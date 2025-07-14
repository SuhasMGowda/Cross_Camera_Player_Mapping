# Cross-Camera Player Mapping using YOLO and DeepSORT

This project maps and tracks players across two different camera angles (`broadcast.mp4` and `tacticam.mp4`) from a football match. It uses:
- YOLOv8/YOLOv11 for player detection
- DeepSORT for tracking
- Feature-based matching to associate players across views

---

## Dependencies

- pip install opencv-python torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
- pip install ultralytics deep-sort-realtime opencv-python tqdm numpy matplotlib

Download best.pt from (https://drive.google.com/file/d/1-5fOSHOSB9UXyP_enOoZNAMScrePVcMD/view) and save it in your directory.

---

## Execution

Go to your current directory 
- Run detect_track.py
- map_players.py
- output.py

---

## Working

- Detect and track players in both videos
- Extract histogram + spatial features
- Match players from tacticam to broadcast
- Save the final visualization

---

## Summary

This project demonstrates cross-view player identification using classic computer vision + deep tracking. While performance is good in ideal frames, improvements can be made using ReID models, temporal alignment, and jersey text OCR for robustness.

---
