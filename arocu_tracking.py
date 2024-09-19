import os
import cv2
import cv2.cuda
import torch
import time
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLOv10
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor
from absl import app, flags
import numpy as np
from typing import Optional
from threading import Thread
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from abc import ABC, abstractmethod
from dataclasses import dataclass
from scipy.spatial.distance import cosine
import logging
#from cam_commun import CameraConnection
from pt_commun import CaptrackConnection
from aruco_detector import ArucoDetector
import warnings
from datetime import datetime


warnings.filterwarnings("ignore", category=Warning)
cuda_stream = torch.cuda.Stream()
# Suppress specific library logs
logging.getLogger("deep_sort_realtime").setLevel(logging.ERROR)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)


def get_adjustment(axis_len, pos) -> (float, int):
    """Returns the adjustment magnitude and direction based on the position of the target marker on the axis."""
    normalised_adjustment = pos / axis_len - 0.5
    adjustment_magnitude = abs(round(normalised_adjustment,1))

    if normalised_adjustment > 0:
        return adjustment_magnitude, -1
    else:
        return adjustment_magnitude, 1

def initialize_model():
    if not os.path.exists(args.model_path):
        logger.error(f"Model weights not found at {args.model_path}")
        raise FileNotFoundError("Model weights file not found")

    model = YOLOv10(args.model_path)

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

        if confidence < args.conf:
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
                if similarity > args.reid:
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

def track_loop(Kp: float, Kd: float, x_direction: int, y_direction: int, g_cam_username, g_cam_pass, g_cam_ip):
    """Loop to continuously track the target marker and adjust the gimbal accordingly."""
    try:
        os.makedirs(f"outputs", exist_ok=True)
        output_video_path = f'outputs/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.mp4'
        print("Tracking loop starting...")
        vid = cv2.VideoCapture(
        "filesrc location=videos/cam0.mp4 ! qtdemux ! h264parse ! nvv4l2decoder ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! appsink",
        cv2.CAP_GSTREAMER)

        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        #vid = BufferedCamera(g_cam_username, g_cam_pass, g_cam_ip)
        gimbal = CaptrackConnection()
        gimbal.connect_to_captrack()
        gimbal.axis_on()

        model = initialize_model()  # Model runs on GPU
        class_names = ['drone']
        tracker = DeepSort(embedder="torchreid", max_age=7, n_init=1, max_cosine_distance=100000, max_iou_distance=0.1)
        color = (92, 179, 102)
        counter = 0
        display_counter = 0
        track_class_mapping = {}
        historical_embeddings = {}
        display_ids_mapping = {}

        frame_counter = 0
        start_time = time.time()
        # cam_connection = CameraConnection()

        x_mag_old = None
        y_mag_old = None

        with ThreadPoolExecutor(max_workers=4) as executor:
            while True:
                ret, frame = vid.read()
                cv2.imshow("image", ret)
                cv2.waitKey(1)
                #detected_targets = detector.detect(img)
                tracks = process_frame_gpu(frame, model, tracker)
                frame, counter, display_counter = draw_tracks(frame, tracks, class_names, color, counter, display_counter, track_class_mapping, historical_embeddings, display_ids_mapping, fps)

                if len(tracks) != 1:
                    # print("Expecting only one marker, got", len(detected_targets))
                    x_mag_old = None
                    continue
                frame_counter += 1
                elapsed_time = time.time() - start_time
                if elapsed_time > 0:
                    fps = frame_counter / elapsed_time

                # Write processed frame to file
                #executor.submit(writer.write, frame)
                writer.write(frame)
                # Show the frame in a real-time display window
                cv2.imshow("Drone Tracking", frame)

                # Get adjustment
                #height, width = frame.shape
                ltrb = tracks[0].to_ltrb()
                x1, y1, x2, y2 = map(int, ltrb)

                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                x_mag, x_dir = get_adjustment(width, cx)
                y_mag, y_dir = get_adjustment(height, cy)

                # first detection in a sequence
                if x_mag_old is None or y_mag_old is None:
                    x_mag_old = x_mag
                    y_mag_old = y_mag

                # Proportional
                adj_Kpx = Kp * x_mag
                adj_Kpy = Kp * y_mag
                #Derivative
                adj_Kdx = Kd * (x_mag - x_mag_old)
                adj_Kdy = Kd * (y_mag - y_mag_old)

                adjustment_x = x_direction * (adj_Kpx + adj_Kdx)
                adjustment_y = y_direction * (adj_Kpy + adj_Kdy)

                gimbal.speed_movement(1, adjustment_x, 1000)
                gimbal.speed_movement(2, adjustment_y, 1000)

                x_mag_old = x_mag
                y_mag_old = y_mag


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
            vid.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        parser = ArgumentParser(description='Control a gimbal and track aruco target', formatter_class=ArgumentDefaultsHelpFormatter)
        parser.add_argument('--Kp', type=float, default=23, help='Proportional gain')
        parser.add_argument('--Kd', type=float, default=1.2, help='Derivative gain')
        parser.add_argument('--x_direction', type=int, choices=[-1, 1], default=-1, help='X axis direction')
        parser.add_argument('--y_direction', type=int, choices=[-1, 1], default=1, help='Y axis direction')
        parser.add_argument('--g_cam_username', type=str, default="admin", help='RTSP camera username')
        parser.add_argument('--g_cam_pass', type=str, default="123456", help='RTSP camera password')
        parser.add_argument('--g_cam_ip', type=str, default="192.168.10.176", help='RTSP camera IP address')
        parser.add_argument('--conf', type=float, default="0.3", help='Threshold for similarity comparison')
        parser.add_argument('--reid', type=float, default="0.5", help='Confidance level')
    
        parser.add_argument('--model_path', type=str, default="/weights/best.pt", help='yolo weights')
        args = parser.parse_args()

        #detector = ArucoDetector()
        track_loop(args.Kp, args.Kd, args.x_direction, args.y_direction, args.g_cam_username, args.g_cam_pass, args.g_cam_ip)
    except SystemExit:
        pass   