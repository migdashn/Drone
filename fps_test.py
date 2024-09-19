import os
import cv2
import torch
import logging
import time
from absl import app, flags
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLOv10
from scipy.spatial.distance import cosine
import warnings

warnings.filterwarnings("ignore", category=Warning)
cuda_stream = torch.cuda.Stream()
# Suppress specific library logs
logging.getLogger("deep_sort_realtime").setLevel(logging.ERROR)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

# Define command line flags
flags.DEFINE_string("video", "videos/cam0.mp4", "Path to input video or webcam index (0)")
flags.DEFINE_string("output", "output", "Path to output video")
flags.DEFINE_float("conf", 0.50, "Confidence threshold")
flags.DEFINE_string("model_path", "weights/best.pt", "Drone detection weights")
flags.DEFINE_float("reid", 0.50, "Threshold for similarity comparison")
flags.DEFINE_string("g_cam_username", "admin", "Camera username")
flags.DEFINE_string("g_cam_pass", "cap5241", "Camera password")
flags.DEFINE_string("g_cam_ip", "192.168.10.108", "Camera IP")
flags.DEFINE_string("input", "live", "live camera or off-line video")

FLAGS = flags.FLAGS

# CUDA stream for parallelizing GPU operations
stream = torch.cuda.Stream()

def initialize_video_capture():
    if FLAGS.input == "live":
        camera_source = f"rtspsrc location=rtsp://{FLAGS.g_cam_username}:{FLAGS.g_cam_pass}@{FLAGS.g_cam_ip}/cam/realmonitor?channel=1&subtype=0 latency=0 ! queue ! rtph264depay ! h264parse ! nvv4l2decoder enable-max-performance=1 ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink drop=1"
        cap = cv2.VideoCapture(camera_source, cv2.CAP_GSTREAMER)
    elif FLAGS.input == "video":
        cap = cv2.VideoCapture(FLAGS.video)
    else:
        logger.error("Invalid input type. Use 'live' for RTSP or 'video' for offline video.")
        raise ValueError("Invalid input type. Use 'live' for RTSP or 'video' for offline video.")
    
    if not cap.isOpened():
        logger.error("Error: Unable to open video source.")
        raise ValueError("Unable to open video source")
    return cap

def initialize_model():
    if not os.path.exists(FLAGS.model_path):
        logger.error(f"Model weights not found at {FLAGS.model_path}")
        raise FileNotFoundError("Model weights file not found")

    model = YOLOv10(FLAGS.model_path)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model.to(device)
    logger.info(f"Using {device} as processing device")
    return model

def process_frame_gpu(frame, model, tracker):
    # Convert frame to a CUDA GpuMat for faster processing
    frame_gpu = cv2.cuda_GpuMat()
    frame_gpu.upload(frame)

    # YOLOv10 inference on the frame using CUDA
    with torch.cuda.stream(cuda_stream):
        results = model(frame)[0]
    
    detections = []
    for det in results.boxes:
        label, confidence, bbox = det.cls, det.conf, det.xyxy[0]
        x1, y1, x2, y2 = map(int, bbox)
        class_id = int(label)

        if confidence < FLAGS.conf:
            continue

        detections.append([[x1, y1, x2 - x1, y2 - y1], confidence, class_id])

    tracks = tracker.update_tracks(detections, frame=frame)
    return tracks

def draw_tracks(frame, tracks, class_names, color, counter, display_counter, track_class_mapping, historical_embeddings, display_ids_mapping, fps):
    # Draw FPS on the frame
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    for track in tracks:
        if track.is_deleted():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        class_id = track.get_det_class()
        x1, y1, x2, y2 = map(int, ltrb)

        # Assign a new class-specific ID if the track_id is seen for the first time
        if track_id not in track_class_mapping:
            counter += 1
            track_class_mapping[track_id] = counter

        # Check for re-identification
        if track_id not in historical_embeddings:
            historical_embeddings[track_id] = []

        current_embedding = track.get_feature()  # Get the current feature (embedding) for the track
        historical_embeddings[track_id].append(current_embedding)

        # Compare with historical embeddings
        for prev_id, embeddings in historical_embeddings.items():
            if prev_id == track_id:
                continue
            for emb in embeddings:
                similarity = 1 - cosine(current_embedding, emb)
                if similarity > FLAGS.reid:
                    track_class_mapping[track_id] = track_class_mapping.get(prev_id, counter + 1)
                    historical_embeddings[track_id] = embeddings
                    break

        class_specific_id = track_class_mapping[track_id]
        if class_specific_id not in display_ids_mapping:
            display_counter += 1
            display_ids_mapping[class_specific_id] = display_counter

        text = f"{display_ids_mapping[class_specific_id]} - {class_names[class_id]}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(frame, (x1 - 1, y1 - 20), (x1 + len(text) * 12, y1), color, -1)
        cv2.putText(frame, text, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return frame, counter, display_counter

def main(_argv):
    try:
        os.makedirs(FLAGS.output, exist_ok=True)

        input_video_path = FLAGS.video
        output_video_path = f'{FLAGS.output}/output1.mp4'

        cap = initialize_video_capture()
        model = initialize_model()
        class_names = ['drone']

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

        tracker = DeepSort(embedder="torchreid", max_age=7, n_init=1, max_cosine_distance=100000, max_iou_distance=0.1)

        color = (92, 179, 102)

        counter = 0
        display_counter = 0
        track_class_mapping = {}
        historical_embeddings = {}
        display_ids_mapping = {}

        # FPS calculation variables
        prev_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Calculate FPS
            curr_time = time.time()
            elapsed_time = curr_time - prev_time
            fps = 1 / elapsed_time
            prev_time = curr_time

            # Process frame using CUDA
            tracks = process_frame_gpu(frame, model, tracker)
            print(tracks)

            # Draw tracks on frame
            frame, counter, display_counter = draw_tracks(frame, tracks, class_names, color, counter, display_counter, track_class_mapping, historical_embeddings, display_ids_mapping, fps)

            writer.write(frame)

            # Show the frame with bounding boxes and track IDs
            cv2.imshow("drone tracking", frame)

            # Exit if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        logger.info("Class counts:")
        logger.info(f"{class_names[0]}: {counter}")

    except Exception as e:
        logger.exception("An error occurred during processing")
    finally:
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        if 'writer' in locals():
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        app.run(main)
    except SystemExit:
        pass

