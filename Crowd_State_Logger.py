from ultralytics import YOLO
import cv2
import time

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

crowd_history = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    person_count = 0

    for r in results:
        for box in r.boxes:
            if int(box.cls) == 0:
                person_count += 1

    # Save timestamp + count
    crowd_history.append({
        "timestamp": time.time(),
        "crowd_count": person_count
    })

    # Display
    cv2.putText(frame, f"People Count: {person_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.imshow("Crowd Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("\nCrowd history recorded:")
for entry in crowd_history[-10:]:  # show last 10 values
    print(entry)
