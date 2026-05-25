import cv2
import numpy as np
import math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

print("🎨 LOADING FIXED RAINBOW PAINTER...")

# 1. Model Setup
model_path = "hand_landmarker.task"
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1, running_mode=vision.RunningMode.IMAGE)
detector = vision.HandLandmarker.create_from_options(options)

# 2. Webcam & Canvas
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
canvas = np.zeros((480, 640, 3), dtype=np.uint8)

# 3. Config
hue = 0
prev_x, prev_y = None, None
mode = "PAINT"
mode_conf = 0  # Debounce counter

print("✅ Index Finger = Draw | Pinch/Fist = Erase")
print("🌈 Auto-Rainbow + Neon Glow | 's'=Save | 'q'=Quit")

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = detector.detect(mp_img)

        #  Auto-Rainbow Color Cycle
        hue = (hue + 2) % 180
        brush_color = tuple(int(c) for c in cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0])

        if res.hand_landmarks:
            hand = res.hand_landmarks[0]
            idx = hand[8]
            thm = hand[4]
            h, w, _ = frame.shape
            
            # Direct coordinates (Responsive drawing)
            x, y = int(idx.x * w), int(idx.y * h)

            #  Pinch Detection
            pinch_dist = math.hypot(idx.x - thm.x, idx.y - thm.y)
            target_mode = "ERASER" if pinch_dist < 0.07 else "PAINT"

            # 🔒 Debounce (Stable mode switching)
            if target_mode == mode:
                mode_conf += 1
            else:
                mode_conf = max(0, mode_conf - 2)
            if mode_conf >= 3:
                mode = target_mode

            # ️ DRAWING LOGIC
            if prev_x is not None:
                if mode == "PAINT":
                    # Neon Effect: Thick color line + thin white core
                    cv2.line(canvas, (prev_x, prev_y), (x, y), brush_color, 10)
                    cv2.line(canvas, (prev_x, prev_y), (x, y), (255, 255, 255), 3)
                elif mode == "ERASER":
                    # Thick black eraser
                    cv2.line(canvas, (prev_x, prev_y), (x, y), (0, 0, 0), 50)
                    
            prev_x, prev_y = x, y
            
            # Tracking Dot (Yellow) so you know exactly where it's drawing
            cv2.circle(frame, (x, y), 6, (0, 255, 255), -1)
        else:
            prev_x, prev_y = None, None

        # ️ Overlay Canvas on Video
        cv2.addWeighted(frame, 0.6, canvas, 0.4, 0, frame)
        
        # Mode Indicator
        cv2.rectangle(frame, (200, 10), (440, 45), (0,0,0), -1)
        cv2.putText(frame, mode, (250, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 
                    (0, 255, 0) if mode=="PAINT" else (0, 0, 255), 2)
        cv2.putText(frame, "Pinch to Erase", (230, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

        cv2.imshow("Rainbow Neon Painter", frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'): break
        elif key == ord('s'):
            fn = "my_rainbow_art.png"
            cv2.imwrite(fn, canvas)
            print(f" Saved: {fn}")

except KeyboardInterrupt:
    pass
finally:
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Stopped!")