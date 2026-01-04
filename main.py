import cv2
from ultralytics import YOLO
import cvzone
from imutils.video import VideoStream
import time

# ------------------ USER CHOICE ------------------
print("Select Video Source:")
print("1. Video file")
print("2. Live CCTV / Webcam")

choice = input("Enter choice (1 or 2): ")

# ------------------ VIDEO SOURCE ------------------
if choice == "1":
    cap = cv2.VideoCapture("vid.mp4")   # video file
    use_videostream = False
else:
    # For webcam
    # vs = VideoStream(src=0).start()

    # For CCTV (RTSP example)
    vs = VideoStream(src="rtsp://admin:admin%401234@41.139.156.38:554/Streaming/Channels/101").start()

    #vs = VideoStream(src=0).start()
    time.sleep(2.0)
    use_videostream = True

# ------------------ YOLO MODEL ------------------
model = YOLO("yolo12n.pt")
names = model.names

# ------------------ VARIABLES ------------------
frame_count = 0
hist = {}
counted = set()
in_count = 0
out_count = 0

line_p1 = (706, 216)
line_p2 = (983, 311)

# ------------------ MOUSE DEBUG ------------------
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        print(f"Mouse: [{x}, {y}]")

cv2.namedWindow("RGB")
cv2.setMouseCallback("RGB", RGB)

# ------------------ LINE SIDE FUNCTION ------------------
def counter(px, py, x1, y1, x2, y2):
    return (x2 - x1)*(py - y1) - (y2 - y1)*(px - x1)

# ------------------ MAIN LOOP ------------------
while True:
    if use_videostream:
        frame = vs.read()
        if frame is None:
            break
    else:
        ret, frame = cap.read()
        if not ret:
            break

    frame_count += 1
    if frame_count % 3 != 0:
        continue

    frame = cv2.resize(frame, (1020, 600))

    results = model.track(frame, persist=True, classes=[0])

    if results[0].boxes.id is not None:
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        class_ids = results[0].boxes.cls.int().cpu().tolist()

        for track_id, box, class_id in zip(ids, boxes, class_ids):
            x1, y1, x2, y2 = box
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            if track_id in hist:
                prev_cx, prev_cy = hist[track_id]
                prev_side = counter(prev_cx, prev_cy, *line_p1, *line_p2)
                curr_side = counter(cx, cy, *line_p1, *line_p2)

                if prev_side * curr_side < 0 and track_id not in counted:
                    if curr_side < 0:
                        in_count += 1
                    else:
                        out_count += 1
                    counted.add(track_id)

            hist[track_id] = (cx, cy)

            cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cvzone.putTextRect(frame, f'ID: {track_id}', (x1, y1), 1, 1)

    cv2.line(frame, line_p1, line_p2, (255, 255, 255), 2)
    cvzone.putTextRect(frame, f'IN: {in_count}', (50, 30), 2, 2, colorR=(0, 255, 0))
    cvzone.putTextRect(frame, f'OUT: {out_count}', (50, 80), 2, 2, colorR=(0, 0, 255))

    cv2.imshow("RGB", frame)

    if cv2.waitKey(0) & 0xFF == 27:
        break

# ------------------ CLEANUP ------------------
if use_videostream:
    vs.stop()
else:
    cap.release()

cv2.destroyAllWindows()

