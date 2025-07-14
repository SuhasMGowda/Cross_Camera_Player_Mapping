from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2, json

import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

def detect_and_track(video_path, tag):
    model = YOLO(r"best.pt")
    model.to(device)

    tracker = DeepSort(max_age=30)
    cap = cv2.VideoCapture(video_path)

    frame_id = 0
    detections = {}

    while True:
        ret, frame = cap.read()
        if not ret: break

        results = model.predict(frame, verbose=False, device=0 if device == "cuda" else "cpu")[0]
        boxes = results.boxes.xyxy.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()

        frame_tracks = []
        for i, box in enumerate(boxes):
            if int(classes[i]) != 2:
                continue
            x1, y1, x2, y2 = map(int, box)
            frame_tracks.append(([x1, y1, x2 - x1, y2 - y1], 0.9, "player"))

        tracks = tracker.update_tracks(frame_tracks, frame=frame)

        for track in tracks:
            if not track.is_confirmed(): continue
            tid = track.track_id
            bbox = track.to_ltrb()
            det = {
                "frame": frame_id,
                "id": tid,
                "bbox": list(map(int, bbox))
            }
            detections[f"{tag}_frame{frame_id}_id{tid}"] = det

        frame_id += 1

    cap.release()
    with open(f"detections_{tag}.json", "w") as f:
        json.dump(detections, f, indent=2)
    print(f"{tag} detections saved!")

detect_and_track("broadcast.mp4", "broadcast")
detect_and_track("tacticam.mp4", "tacticam")
