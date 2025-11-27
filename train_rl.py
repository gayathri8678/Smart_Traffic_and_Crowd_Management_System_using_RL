# train_rl.py
import cv2
import csv
import time
import argparse
from collections import deque
from rl_agent import QLearningAgent
from reward import reward_function

# You must implement or import detect_people(frame) in this file or import from Crowd_Detection.py
# For example, if your Crowd_Detection.py exposes a detect_people function you can import it:
# from Crowd_Detection import detect_people

# For standalone demonstration, here's a placeholder simple detector wrapper:
def detect_people_from_frame(frame, model=None, conf=0.65):
    """
    If you have a 'model' available (YOLO), pass it. Otherwise, implement your own.
    This placeholder returns 0 always - replace with your real detection call.
    """
    if model is None:
        # dummy: replace with actual model inference
        return 0
    # If you pass an ultralytics YOLO model, it returns results = model(frame, conf=conf, classes=[0])
    results = model(frame, conf=conf, iou=0.5, classes=[0])
    return len(results[0].boxes)

def run_training(mode="simulate", video_path=None, episodes=200, steps_per_episode=200, save_prefix="logs"):
    agent = QLearningAgent(max_capacity=10)
    # We'll log per-episode total reward and step-level CSV for deeper analysis
    episode_log = []
    csvfile = open(f"{save_prefix}_step_logs.csv", "w", newline="")
    writer = csv.writer(csvfile)
    writer.writerow(["episode","step","timestamp","people_count","action","reward","epsilon"])

    # Simple synthetic simulator: create sequences where crowd grows, stays, then drops
    import numpy as np
    for ep in range(episodes):
        total_reward = 0
        # create a synthetic sequence if mode==simulate
        if mode == "simulate":
            # example pattern: ramp-up, hold, ramp-down with noise
            ramp_up = np.clip(np.linspace(0, 8, steps_per_episode//3) + np.random.randint(-1,2, steps_per_episode//3), 0, 10)
            hold = np.clip(np.ones(steps_per_episode//3) * 8 + np.random.randint(-1,2, steps_per_episode//3), 0, 10)
            ramp_down = np.clip(np.linspace(8, 0, steps_per_episode - 2*(steps_per_episode//3)) + np.random.randint(-1,2, steps_per_episode - 2*(steps_per_episode//3)), 0, 10)
            seq = np.concatenate([ramp_up, hold, ramp_down]).astype(int)
        else:
            # webcam or video: we will read frames
            cap = cv2.VideoCapture(0 if mode=="webcam" else video_path)
            seq = None  # we'll read in steps

        prev_count = 0
        for step in range(steps_per_episode):
            # get people_count
            if mode == "simulate":
                people_count = int(seq[step])
            else:
                ret, frame = cap.read()
                if not ret:
                    break
                # optionally include your YOLO model here; for now we assume detect_people_from_frame uses a model
                people_count = detect_people_from_frame(frame, model=None, conf=0.75)

            # choose action
            action = agent.choose_action(people_count)
            # compute reward; reward_function expects (current_count, previous_count, action)
            reward = reward_function(people_count, previous_count := prev_count, action=action)
            total_reward += reward

            # for RL update we will assume next_count == current_count for now (or for simulation can peek next)
            # if simulate mode, we can peek next
            next_count = people_count
            if mode == "simulate" and step < (steps_per_episode - 1):
                next_count = int(seq[step + 1])

            # learn
            agent.learn(prev_count, action, reward, next_count)

            # log step
            writer.writerow([ep, step, time.time(), people_count, action, reward, agent.epsilon])
            prev_count = people_count

        if mode != "simulate" and 'cap' in locals():
            cap.release()

        episode_log.append((ep, total_reward))
        print(f"Episode {ep}/{episodes-1} total_reward={total_reward:.2f} epsilon={agent.epsilon:.3f}")

        # occasional save
        if (ep + 1) % 20 == 0:
            agent.save(f"{save_prefix}_qtable_ep{ep+1}.json")

    csvfile.close()
    # save final
    agent.save(f"{save_prefix}_qtable_final.json")
    # save episode summary
    with open(f"{save_prefix}_episode_summary.csv","w",newline="") as ef:
        w = csv.writer(ef)
        w.writerow(["episode","total_reward"])
        for e,r in episode_log:
            w.writerow([e,r])

    print("Training finished. Logs and Q-table saved.")
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["simulate","webcam","video"], default="simulate")
    parser.add_argument("--video", default=None)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--steps", type=int, default=120)
    args = parser.parse_args()
    run_training(mode=args.mode, video_path=args.video, episodes=args.episodes, steps_per_episode=args.steps)
