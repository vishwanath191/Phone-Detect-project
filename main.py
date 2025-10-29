import cv2
import time
import torch
from datetime import datetime
import numpy as np
import os

# Parameters
CONF_THRESHOLD = 0.4   # detection confidence threshold
SAVE_ON_DETECT = True  # whether to save a snapshot when phone detected
OUTPUT_DIR = "phone_captures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load YOLOv5 model from torch.hub (ultralytics/yolov5). Uses 'yolov5s' by default (small and fast).
# This requires internet the first time to download weights.
print("Loading YOLOv5 model (this may take a moment the first time)...")
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.conf = CONF_THRESHOLD  # set model confidence threshold

# The model returns names for classes. We'll check for 'cell phone' or similar.
class_names = model.names  # dict: idx -> name
# Look up candidate names that indicate a phone
phone_name_candidates = {'cell phone', 'mobile phone', 'phone', 'cellphone', 'cell_phone'}

# Try to find which class index corresponds to phone (if any)
phone_class_indices = [i for i, n in class_names.items() if any(c in n.lower() for c in phone_name_candidates)]
print("YOLO class names sample (first 20):", {i:n for i,n in list(class_names.items())[:20]})
print("Detected phone class indices in model names:", phone_class_indices)
if not phone_class_indices:
    print("Warning: Model doesn't list an explicit 'phone' class name. The model may still detect phones under different labels.")
    # We will still filter by label text when parsing results.

# Open webcam (default device 0)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam. Check camera permissions and device index.")

print("Webcam opened. Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame from webcam. Exiting.")
        break

    # Convert BGR -> RGB for model
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Inference (returns a special object with .xyxy, .pandas(), .pandas().xyxy etc.)
    results = model(img_rgb, size=640)  # you can change size (320 / 640 / 1280)
    # results.xyxy[0] gives tensor of [x1, y1, x2, y2, conf, cls]
    detections = results.xyxy[0].cpu().numpy() if len(results.xyxy) > 0 else np.array([])

    phone_count = 0
    # Optionally use pandas result for label text
    # df = results.pandas().xyxy[0]  # pandas dataframe with columns: xmin,ymin,xmax,ymax,confidence,class,name

    # Use results.pandas for easy access to label names
    df = results.pandas().xyxy[0]

    # Process detections
    for _, det in df.iterrows():
        xmin = int(det['xmin'])
        ymin = int(det['ymin'])
        xmax = int(det['xmax'])
        ymax = int(det['ymax'])
        conf = float(det['confidence'])
        label = str(det['name']).lower()

        # Decide if this detection is a phone
        is_phone = False
        # If model provided one of our known phone-like class names:
        if any(c in label for c in phone_name_candidates):
            is_phone = True
        # Otherwise optionally check label text containing 'phone' substring
        if 'phone' in label or 'cell' in label:
            is_phone = True

        if is_phone and conf >= CONF_THRESHOLD:
            phone_count += 1
            # draw bounding box
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            text = f"{det['name']} {conf:.2f}"
            cv2.putText(frame, text, (xmin, max(ymin-6, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    # Display count on frame
    cv2.putText(frame, f"Phones: {phone_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)

    # If detection happened, save snapshot (one per frame where present)
    if phone_count > 0 and SAVE_ON_DETECT:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        fname = os.path.join(OUTPUT_DIR, f"phone_capture_{ts}.jpg")
        cv2.imwrite(fname, frame)
        # Terminal bell (may or may not produce sound depending on platform)
        print("\a", end="")  # attempt beep
        print(f"[{datetime.now().isoformat()}] Phone detected! Saved {fname}")

    # Show frame
    cv2.imshow("Phone Detector (press q to quit)", frame)

    # Break on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("Finished.")
