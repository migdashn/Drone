import os
import cv2
import torch
import logging
import time
import concurrent.futures
from absl import app, flags
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLOv10
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define command line flags
flags.DEFINE_string("videos", "videos", "Path to input videos or webcam index (0)")
flags.DEFINE_string("output", "output", "Path to output video")
flags.DEFINE_float("conf", 0.40, "Confidence threshold")
flags.DEFINE_string("model_path", "weights/best.pt", "Drone detection weights")
flags.DEFINE_string("g_cam_username", "admin", "Camera username")
flags.DEFINE_string("g_cam_pass", "cap5241", "Camera password")
flags.DEFINE_string("g_cam_ip", "192.168.10.108", "Camera IP")

FLAGS = flags.FLAGS

def initialize_video_capture():
    # RTSP camera stream setup using GStreamer for low latency
    camera_source = f"rtspsrc location=rtsp://{FLAGS.g_cam_username}:{FLAGS.g_cam_pass}@{FLAGS.g_cam_ip}/cam/realmonitor?channel=1&subtype=0 latency=0 ! queue ! rtph264depay ! h264parse ! nvv4l2decoder enable-max-performance=1 ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink drop=1"
    #camera_source = f"rtspsrc location=rtsp://{FLAGS.g_cam_username}:{FLAGS.g_cam_pass}@{FLAGS.g_cam_ip}/cam/realmonitor?channel=1&subtype=0 latency=0 ! queue max-size-buffers=1 max-size-bytes=0 max-size-time=0 ! rtph264depay ! h264parse ! nvv4l2decoder enable-max-performance=1 ! nvvidconv ! video/x-raw,format=BGR,width=1920,height=1080 ! appsink drop=1"

    cap = cv2.VideoCapture(camera_source, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        logger.error("Error: Unable to open RTSP video stream.")
        raise ValueError("Unable to open video source")

    return cap

def initialize_model():
    if not os.path.exists(FLAGS.model_path):
        logger.error(f"Model weights not found at {FLAGS.model_path}")
        raise FileNotFoundError("Model weights file not found")

    model = YOLOv10(FLAGS.model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info(f"Using {device} as processing device")
    return model

def process_frame(frame, model, tracker, device):
    # Resize the frame to a compatible size (640x640 in this case)
    frame_resized = cv2.resize(frame, (640, 640))

    # Convert the frame to a PyTorch tensor and move it to the GPU
    frame_tensor = torch.from_numpy(frame_resized).to(device)

    # Transpose the frame to match the BCHW format (Batch, Channels, Height, Width)
    frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0).float() / 255.0  # Normalize to [0, 1]

    # Run inference on the GPU
    results = model(frame_tensor, verbose=False)[0]

    detections = []
    for det in results.boxes:
        label, confidence, bbox = det.cls, det.conf, det.xyxy[0]
        x1, y1, x2, y2 = map(int, bbox)
        class_id = int(label)

        if confidence < FLAGS.conf:
            continue

        detections.append([[x1, y1, x2 - x1, y2 - y1], confidence, class_id])

    # Run tracking on the CPU
    tracks = tracker.update_tracks(detections, frame=frame)
    return tracks



def draw_tracks(frame, tracks, class_names, color, counter, track_class_mapping, fps):
    # Draw the FPS on the top left corner
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        class_id = track.get_det_class()
        x1, y1, x2, y2 = map(int, ltrb)

        # Assign a new class-specific ID if the track_id is seen for the first time
        if track_id not in track_class_mapping:
            counter += 1
            track_class_mapping[track_id] = counter

        class_specific_id = track_class_mapping[track_id]
        text = f"{class_specific_id} - {class_names[class_id]}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(frame, (x1 - 1, y1 - 20), (x1 + len(text) * 12, y1), color, -1)
        cv2.putText(frame, text, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return frame, counter

def main(_argv):
    try:
        os.makedirs(FLAGS.output, exist_ok=True)

        # Initialize RTSP video capture
        cap = initialize_video_capture()
        model = initialize_model()  # Model runs on GPU
        class_names = ['drone']

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        logger.info(f"Video Frame Dimensions: {frame_width}x{frame_height} at {fps} FPS")

        if fps == 0:
            fps = 50

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        output_video_path = f'{FLAGS.output}/processed_output_video.mp4'
        writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
        output_clean_path = f'{FLAGS.output}/clean_output_video.mp4'
        clean_writer = cv2.VideoWriter(output_clean_path, fourcc, fps, (frame_width, frame_height))

        tracker = DeepSort(max_age=5, n_init=1, max_cosine_distance=0.4, max_iou_distance=0.2)

        color = (92, 179, 102)
        counter = 0
        track_class_mapping = {}

        frame_counter = 0
        fps = 0
        start_time = time.time()

        # Use ThreadPoolExecutor to handle video writing tasks in parallel
        with ThreadPoolExecutor(max_workers=2) as executor:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.error("Failed to grab frame from video stream.")
                    break

                clean_frame = frame.copy()

                # Offload video writing tasks to a thread
                executor.submit(clean_writer.write, clean_frame)  # Write clean frame to file

                # Process the frame on GPU (YOLO) and track on CPU (DeepSORT)
                tracks = process_frame(frame, model, tracker, torch.device("cuda"))
                
                # Draw tracked bounding boxes and calculate FPS
                frame, counter = draw_tracks(frame, tracks, class_names, color, counter, track_class_mapping, fps)

                frame_counter += 1
                elapsed_time = time.time() - start_time
                if elapsed_time > 0:
                    fps = frame_counter / elapsed_time

                # Write processed frame to file
                executor.submit(writer.write, frame)

                # Show the frame in a real-time display window
                cv2.imshow("Drone Tracking", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("Exiting and saving videos...")
                    break

    except KeyboardInterrupt:
        logger.info("Script terminated with CTRL + C. Saving videos...")

    except Exception as e:
        logger.exception("An error occurred during processing")

    finally:
        logger.info("Releasing resources and saving videos.")
        if 'cap' in locals():
            cap.release()
        if 'writer' in locals():
            writer.release()
        if 'clean_writer' in locals():
            clean_writer.release()
        cv2.destroyAllWindows()



if __name__ == "__main__":
    try:
        app.run(main)
    except SystemExit:
        pass

