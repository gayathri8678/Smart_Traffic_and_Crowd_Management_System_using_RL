<<<<<<< HEAD

=======
from ultralytics import YOLO
import cv2
from rl_agent import RLAgent
from reward import reward_function

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

agent = RLAgent()
prev_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    results = model(frame, verbose=False)

    # Count people
    people_count = 0
    for r in results:
        for box in r.boxes.cls:
            if int(box) == 0:  # class 0 = person
                people_count += 1

    # ---- RL logic ----
  action = agent.choose_action(people_count)
    reward = reward_function(people_count)
    agent.learn(prev_count, action, reward, people_count)
    prev_count = people_count

    # Show on screen
    cv2.putText(frame, f"People: {people_count}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.putText(frame, f"Action: {action}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
    cv2.putText(frame, f"Reward: {reward}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow("Crowd RL System", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
import winsound

def alert_status(action):
    if action == 1:  # Warning Level
        print("⚠ WARNING: Crowd Density Increasing!")
        winsound.Beep(1000, 500)   # Medium beep
    elif action == 2:  # Emergency
        print("🚨 EMERGENCY: OVERCROWDING DETECTED!")
        winsound.Beep(2000, 800)   # High alert beep
>>>>>>> b10dd52 (added rl_agent.py,Crowd_Detection.py,Crowd_State_Logger.py,csv,json,train_rl.py,visualizee.py,yolov8n.pt)
