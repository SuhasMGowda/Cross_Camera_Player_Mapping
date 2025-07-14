import json
import numpy as np

# === Load detections ===
with open("detections_broadcast.json") as f1, open("detections_tacticam.json") as f2:
    broadcast = json.load(f1)
    tacticam = json.load(f2)

# === Extract player features from bbox ===
def extract_features(det):
    x1, y1, x2, y2 = det["bbox"]
    center = [(x1 + x2) / 2, (y1 + y2) / 2]
    size = [(x2 - x1), (y2 - y1)]
    return center, size

# === Mapping tacticam players to broadcast IDs ===
mapping = {}
used_broadcast = set()

for tac_key, tac_det in tacticam.items():
    best_score = float('inf')
    best_broad_key = None

    tac_center, tac_size = extract_features(tac_det)

    for broad_key, broad_det in broadcast.items():
        if broad_key in used_broadcast:
            continue

        broad_center, broad_size = extract_features(broad_det)

        # --- Distance metrics ---
        pos_dist = np.linalg.norm(np.array(tac_center) - np.array(broad_center))
        size_dist = np.linalg.norm(np.array(tac_size) - np.array(broad_size))

        # Weighted total score
        score = 0.7 * pos_dist + 0.3 * size_dist

        if score < best_score:
            best_score = score
            best_broad_key = broad_key

    if best_broad_key:
        mapping[tac_key] = best_broad_key
        used_broadcast.add(best_broad_key)

# === Save mapping ===
with open("player_mapping.json", "w") as f:
    json.dump(mapping, f, indent=2)

print("Player mapping completed and saved to player_mapping.json")
