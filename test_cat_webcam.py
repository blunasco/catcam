import cv2
import os
import time
from datetime import datetime
from pathlib import Path
from notifier import notify_cat
# --- Config ---
CAT_CASCADE_PATH = os.getenv("CAT_CASCADE_PATH", "haarcascade_frontalcatface.xml")
USE_ROI_FOR_CATS = True        # scan only lower half of frame for cats (helps reduce human FP)
PERSIST_FRAMES = 3             # require cat detection in N consecutive frames
COOLDOWN_SEC = 10              # seconds between saved snapshots
CAT_PARAMS = dict(scaleFactor=1.02, minNeighbors=5, minSize=(80, 80))
HUMAN_PARAMS = dict(scaleFactor=1.10, minNeighbors=6,  minSize=(80, 80))
HUMAN_IOU_THRESH = 0.2         # overlap threshold to veto a cat box

# --- Setup ---
MEDIA_DIR = Path("media")
MEDIA_DIR.mkdir(exist_ok=True)

cat_cascade = cv2.CascadeClassifier(CAT_CASCADE_PATH)
if cat_cascade.empty():
    raise RuntimeError(f"Could not load cat cascade at {CAT_CASCADE_PATH}")

# Use OpenCV-bundled human face cascade
human_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
human_cascade = cv2.CascadeClassifier(human_cascade_path)
if human_cascade.empty():
    raise RuntimeError(f"Could not load human face cascade at {human_cascade_path}")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam (index 0)")

print("✅ Webcam opened. Press 'q' to quit.")

consec_cat_frames = 0
last_snapshot_ts = 0.0

def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    area_a = aw * ah
    area_b = bw * bh
    return inter / float(area_a + area_b - inter)

while True:
    ok, frame = cap.read()
    if not ok:
        print("❌ Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 1) Human faces on full frame
    faces = human_cascade.detectMultiScale(gray, **HUMAN_PARAMS)

    # 2) Cats (optionally restrict to lower half ROI)
    if USE_ROI_FOR_CATS:
        h = gray.shape[0]
        y0 = int(h * 0.45)  # scan bottom 55% only
        roi = gray[y0:, :]
        cats_raw = cat_cascade.detectMultiScale(roi, **CAT_PARAMS)
        # offset ROI boxes back to full-frame coords
        cats = [(x, y + y0, w, h_) for (x, y, w, h_) in cats_raw]
    else:
        cats = cat_cascade.detectMultiScale(gray, **CAT_PARAMS)

    # 3) Veto: remove cat boxes that overlap a human face
    cats = [c for c in cats if all(iou(c, f) < HUMAN_IOU_THRESH for f in faces)]

    # 4) Persistence
    if len(cats) > 0:
        consec_cat_frames += 1
    else:
        consec_cat_frames = 0

    # 5) Draw (debug)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)   # red = human
    for (x, y, w, h) in cats:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)   # green = cat

    # 6) Snapshot on confirmed cat + cooldown
    now = time.time()
    if consec_cat_frames >= PERSIST_FRAMES and (now - last_snapshot_ts) >= COOLDOWN_SEC and best_cat is not None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Save full-frame evidence
        full_out = MEDIA_DIR / f"cat_{ts}.jpg"
        cv2.imwrite(str(full_out), frame)

        # Optional: also save a cropped cat patch for quick viewing
        x, y, w, h = best_cat
        cat_crop = frame[max(y,0):y+h, max(x,0):x+w].copy()
        crop_out = MEDIA_DIR / f"cat_{ts}_crop.jpg"
        if cat_crop.size > 0:
            cv2.imwrite(str(crop_out), cat_crop)

        print(f"📸 Cat snapshot saved: {full_out}")
        if cat_crop.size > 0:
            print(f"📎 Cat crop saved: {crop_out}")

        last_snapshot_ts = now
        consec_cat_frames = 0  # reset persistence

        # Notify (best-effort; don't crash on failures)
        try:
            notify_cat(camera_id="local_cam", image_path=full_out, confidence=round(best_conf or 0.0, 4))
        except Exception as e:
            print(f"⚠️ Notify failed: {e}")


    # Show window if GUI is available (no-op if headless install)
    if hasattr(cv2, "imshow"):
        cv2.imshow("Cat Detector (human veto)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
