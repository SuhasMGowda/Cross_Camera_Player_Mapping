import cv2
import json

# Load mapping and detection files
with open("player_mapping.json") as f:
    mapping = json.load(f)

with open("detections_broadcast.json") as f:
    broadcast = json.load(f)

with open("detections_tacticam.json") as f:
    tacticam = json.load(f)

def draw_boxes(video_path, tag, detections):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(f"mapped_output_{tag}.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    frame_num = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        for key, det in detections.items():
            # Key format: e.g., 'broadcast_frame4_id5'
            parts = key.split("_")
            if len(parts) != 3:
                continue

            k_tag, k_frame_str, k_id_str = parts
            if k_tag != tag:
                continue

            try:
                k_frame = int(k_frame_str.replace("frame", ""))
            except:
                continue

            if k_frame != frame_num:
                continue

            # Draw bounding box
            box = list(map(int, det["bbox"]))  # ensure integers
            x1, y1, x2, y2 = box
            pid = det["id"]

            # Get mapped ID if tacticam
            if tag == "tacticam":
                crop_key = f"{tag}_frame{frame_num}_id{pid}"
                mapped_key = mapping.get(crop_key)
                mapped_id = mapped_key.split("_id")[1] if mapped_key else "N/A"
            else:
                mapped_id = str(pid)

            label = f"ID: {mapped_id}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        out.write(frame)
        frame_num += 1

    cap.release()
    out.release()
    print(f"Saved: mapped_output_{tag}.mp4")

# Run for both videos
draw_boxes("broadcast.mp4", "broadcast", broadcast)
draw_boxes("tacticam.mp4", "tacticam", tacticam)