"""
Fast / Modified RL Model Training System for Traffic & Crowd Detection
Speed-focused changes:
 - input resolution reduced to 84x84
 - smaller CNN & decision heads
 - smaller replay buffer and batch size
 - GPU-aware
 - graceful KeyboardInterrupt save
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import gymnasium as gym
from gymnasium import spaces
import cv2
import matplotlib.pyplot as plt
import os
from datetime import datetime
import time

# ----------------- CONFIG -----------------
IMAGE_SIZE = 84        # reduced from 224 -> 84 for speed
REPLAY_CAPACITY = 10000
BATCH_SIZE = 16
DEFAULT_EPISODES = 100  # lower default for quick tests
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ------------------------------------------

# ==================== NEURAL NETWORKS ====================

class CNN_FeatureExtractor(nn.Module):
    """Compact CNN for extracting features from video frames (faster)."""
    def __init__(self, in_channels=3):
        super(CNN_FeatureExtractor, self).__init__()
        # smaller channel counts and kernels suitable for 84x84
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=8, stride=4),  # -> ~ (16, 20, 20)
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),           # -> ~ (32, 9, 9)
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),           # -> ~ (64, 7, 7)
            nn.ReLU(),
            nn.BatchNorm2d(64),
        )
        
    def forward(self, x):
        return self.conv(x)


class DQN_Network(nn.Module):
    """Smaller Deep Q-Network for Decision Making (faster)."""
    def __init__(self, n_actions, device=DEVICE):
        super(DQN_Network, self).__init__()
        self.device = device
        self.feature_extractor = CNN_FeatureExtractor()
        
        # Calculate flattened feature size robustly
        with torch.no_grad():
            test_input = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE)
            conv_output = self.feature_extractor(test_input)
            self.feature_size = conv_output.reshape(1, -1).size(1)
        
        # Smaller decision head (fewer params)
        self.decision_network = nn.Sequential(
            nn.Linear(self.feature_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, n_actions)
        )
        
    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.reshape(features.size(0), -1)
        return self.decision_network(features)


class ActorCritic_Network(nn.Module):
    """Compact Actor-Critic Network (if used later)."""
    def __init__(self, n_actions, device=DEVICE):
        super(ActorCritic_Network, self).__init__()
        self.device = device
        self.feature_extractor = CNN_FeatureExtractor()
        
        with torch.no_grad():
            test_input = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE)
            conv_output = self.feature_extractor(test_input)
            self.feature_size = conv_output.reshape(1, -1).size(1)
        
        self.actor = nn.Sequential(
            nn.Linear(self.feature_size, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
            nn.Softmax(dim=-1)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(self.feature_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.reshape(features.size(0), -1)
        return self.actor(features), self.critic(features)


# ==================== ENVIRONMENT ====================

class DetectionEnvironment(gym.Env):
    """Universal Environment for Traffic and Crowd Detection (faster resized frames)"""
    
    def __init__(self, mode='synthetic'):
        super(DetectionEnvironment, self).__init__()
        self.mode = mode  # 'synthetic', 'traffic', 'crowd'
        
        # Action space: 0=safe,1=warning,2=alert,3=critical,4=emergency
        self.action_space = spaces.Discrete(5)
        
        # Observation space: RGB image reduced to IMAGE_SIZE
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8
        )
        
        self.current_frame = None
        self.step_count = 0
        self.max_steps = 200   # reduce per-episode steps for quick runs
        self.density = 0.0
        self.risk_level = 0.0
        self.objects_detected = []
        # lighter background subtractor params for speed
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=200, varThreshold=25, detectShadows=False
        )
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.density = 0.0
        self.risk_level = 0.0
        self.current_frame = self._generate_scenario()
        obs = self._preprocess_frame(self.current_frame)
        return obs, {}
    
    def step(self, action):
        self.step_count += 1
        # Generate new frame (cheaper)
        self.current_frame = self._generate_scenario()
        self.objects_detected = self._detect_objects(self.current_frame)
        self.density = self._calculate_density(self.objects_detected)
        self.risk_level = self._calculate_risk(self.objects_detected)
        reward = self._calculate_reward(action)
        obs = self._preprocess_frame(self.current_frame)
        terminated = self.step_count >= self.max_steps
        truncated = False
        info = {
            'objects': len(self.objects_detected),
            'density': self.density,
            'risk': self.risk_level,
            'action': action
        }
        return obs, reward, terminated, truncated, info
    
    def _generate_scenario(self):
        """Generate synthetic scenario (same logic, faster ops)"""
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 30
        cv2.rectangle(frame, (50, 100), (590, 380), (60, 60, 60), -1)
        num_objects = np.random.randint(5, 20)  # slightly fewer objects
        
        for i in range(num_objects):
            x = np.random.randint(60, 580)
            y = np.random.randint(110, 370)
            if self.mode == 'crowd' or np.random.random() > 0.5:
                w, h = 12, 24   # smaller person rectangles (faster detect)
                color = (np.random.randint(120, 255), np.random.randint(120, 255), np.random.randint(120, 255))
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, -1)
            else:
                w, h = np.random.randint(40, 80), np.random.randint(24, 40)
                color = (np.random.randint(120, 255), np.random.randint(50, 150), np.random.randint(50, 150))
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, -1)
        noise = np.random.randint(0, 16, frame.shape, dtype=np.uint8)
        frame = cv2.add(frame, noise)
        return frame
    
    def _detect_objects(self, frame):
        fg_mask = self.bg_subtractor.apply(frame)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        objects = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 200:  # lower threshold but faster kernel
                x, y, w, h = cv2.boundingRect(contour)
                objects.append({'x': x, 'y': y, 'w': w, 'h': h, 'area': area, 'center': (x + w//2, y + h//2)})
        return objects
    
    def _calculate_density(self, objects):
        if len(objects) == 0:
            return 0.0
        total_area = sum(obj['area'] for obj in objects)
        frame_area = 640 * 480
        return min(total_area / frame_area, 1.0)
    
    def _calculate_risk(self, objects):
        if len(objects) < 2:
            return 0.0
        risk = 0.0
        for i, obj1 in enumerate(objects):
            for obj2 in objects[i+1:]:
                dx = obj1['center'][0] - obj2['center'][0]
                dy = obj1['center'][1] - obj2['center'][1]
                distance = np.sqrt(dx**2 + dy**2)
                if distance < 80:
                    risk += (80 - distance) / 80
        return min(risk / max(len(objects), 1), 1.0)
    
    def _calculate_reward(self, action):
        if self.risk_level > 0.8 or self.density > 0.7:
            appropriate_action = 4
        elif self.risk_level > 0.6 or self.density > 0.5:
            appropriate_action = 3
        elif self.risk_level > 0.4 or self.density > 0.3:
            appropriate_action = 2
        elif self.risk_level > 0.2 or self.density > 0.15:
            appropriate_action = 1
        else:
            appropriate_action = 0
        if action == appropriate_action:
            reward = 10.0
        elif abs(action - appropriate_action) == 1:
            reward = 5.0
        elif abs(action - appropriate_action) == 2:
            reward = 0.0
        else:
            reward = -5.0
        if self.risk_level > 0.7 and action >= 3:
            reward += 5.0
        return reward
    
    def _preprocess_frame(self, frame):
        resized = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))
        return resized
    
    def render(self):
        if self.current_frame is not None:
            display = self.current_frame.copy()
            for obj in self.objects_detected:
                cv2.rectangle(display, (obj['x'], obj['y']), (obj['x']+obj['w'], obj['y']+obj['h']), (0, 255, 0), 2)
            cv2.imshow('Environment', display)
            cv2.waitKey(1)


# ==================== REPLAY BUFFER ====================

class ReplayBuffer:
    """Experience Replay Buffer (stacked numpy arrays on sample)."""
    def __init__(self, capacity=REPLAY_CAPACITY):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        # state & next_state are numpy arrays (H,W,3)
        self.buffer.append((state, action, reward, next_state, float(done)))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.stack(states).astype(np.uint8),
                np.array(actions, dtype=np.int64),
                np.array(rewards, dtype=np.float32),
                np.stack(next_states).astype(np.uint8),
                np.array(dones, dtype=np.float32))
    
    def __len__(self):
        return len(self.buffer)


# ==================== RL AGENT ====================

class DQN_Agent:
    """DQN Agent for Training (faster config)."""
    def __init__(self, n_actions=5, device=DEVICE):
        self.device = device
        self.n_actions = n_actions
        
        # Networks
        self.policy_net = DQN_Network(n_actions, device=device).to(self.device)
        self.target_net = DQN_Network(n_actions, device=device).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-4)
        self.replay_buffer = ReplayBuffer(REPLAY_CAPACITY)
        
        # Hyperparameters
        self.batch_size = BATCH_SIZE
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.02
        self.epsilon_decay = 0.995
        self.target_update_freq = 5  # more frequent updates (smaller model)
    
    def select_action(self, state, training=True):
        # state: numpy HWC uint8
        if training and random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        
        with torch.no_grad():
            state_t = torch.from_numpy(state).float().permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0
            q_values = self.policy_net(state_t)
            return int(q_values.argmax().item())
    
    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return 0.0
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        # convert to tensors and move to device
        states = torch.from_numpy(states).float().permute(0, 3, 1, 2).to(self.device) / 255.0
        next_states = torch.from_numpy(next_states).float().permute(0, 3, 1, 2).to(self.device) / 255.0
        actions = torch.from_numpy(actions).long().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        dones = torch.from_numpy(dones).float().to(self.device)
        
        # current Q
        q_values = self.policy_net(states)
        current_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # target Q
        with torch.no_grad():
            next_q_values = self.target_net(next_states)
            next_max_q, _ = next_q_values.max(dim=1)
            target_q = rewards + (1.0 - dones) * self.gamma * next_max_q
        
        loss = nn.MSELoss()(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        return float(loss.item())
    
    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# ==================== TRAINING FUNCTION ====================

def train_rl_model(episodes=DEFAULT_EPISODES, save_path='models/trained_model.pth'):
    """Main training loop (faster defaults + graceful exit)."""
    print("="*60)
    print("REINFORCEMENT LEARNING TRAINING SYSTEM (FAST MODE)")
    print(f"IMAGE_SIZE: {IMAGE_SIZE}x{IMAGE_SIZE} | DEVICE: {DEVICE}")
    print("="*60)
    
    env = DetectionEnvironment(mode='synthetic')
    agent = DQN_Agent(n_actions=5, device=DEVICE)
    
    print("Environment and agent created.")
    print(f"Starting training for {episodes} episodes...")
    print("="*60)
    
    episode_rewards = []
    episode_losses = []
    risk_accuracy = []
    best_reward = -float('inf')
    
    start_time = time.time()
    try:
        for episode in range(episodes):
            state, _ = env.reset()
            episode_reward = 0.0
            episode_loss = 0.0
            correct_actions = 0
            total_steps = 0
            
            done = False
            while not done:
                action = agent.select_action(state, training=True)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                agent.replay_buffer.push(state, action, reward, next_state, done)
                loss = agent.train_step()
                
                episode_reward += reward
                episode_loss += loss
                if reward > 0:
                    correct_actions += 1
                total_steps += 1
                
                state = next_state
            
            # Target update & epsilon decay
            if episode % agent.target_update_freq == 0:
                agent.update_target()
            agent.decay_epsilon()
            
            # Metrics
            episode_rewards.append(episode_reward)
            episode_losses.append(episode_loss / max(total_steps, 1))
            accuracy = correct_actions / total_steps if total_steps > 0 else 0.0
            risk_accuracy.append(accuracy)
            
            # Save best
            if episode_reward > best_reward:
                best_reward = episode_reward
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save({
                    'model_state_dict': agent.policy_net.state_dict(),
                    'episode': episode,
                    'best_reward': best_reward,
                    'epsilon': agent.epsilon
                }, save_path)
            
            # Logging less frequently
            if (episode + 1) % 10 == 0 or episode == 0:
                avg_reward = np.mean(episode_rewards[-100:]) if episode_rewards else 0.0
                avg_accuracy = np.mean(risk_accuracy[-100:]) if risk_accuracy else 0.0
                elapsed = time.time() - start_time
                print(f"Ep {episode+1:3d}/{episodes} | R:{episode_reward:6.1f} | Avg:{avg_reward:6.2f} | Acc:{avg_accuracy*100:5.1f}% | ε:{agent.epsilon:.3f} | t:{elapsed:.0f}s")
            
            # periodic checkpoint
            if (episode + 1) % 50 == 0:
                chk = f"models/checkpoint_ep{episode+1}.pth"
                torch.save(agent.policy_net.state_dict(), chk)
                print(f"  -> checkpoint saved: {chk}")
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving model...")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(agent.policy_net.state_dict(), save_path)
        print(f"Model saved to {save_path}.")
        return agent
    
    # End training
    total_time = time.time() - start_time
    print("\n" + "="*60)
    print("TRAINING COMPLETED")
    print(f"Total time: {total_time:.1f}s | Best reward: {best_reward:.2f}")
    print("="*60)
    
    # Save final model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(agent.policy_net.state_dict(), save_path)
    print(f"Final model saved to {save_path}")
    
    # Plot results (will be fast for small episode counts)
    try:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.plot(episode_rewards, alpha=0.6); plt.title("Episode Rewards"); plt.grid(True)
        plt.subplot(1, 3, 2)
        plt.plot(episode_losses, alpha=0.6); plt.title("Episode Loss"); plt.grid(True)
        plt.subplot(1, 3, 3)
        plt.plot([acc*100 for acc in risk_accuracy], alpha=0.6); plt.title("Accuracy (%)"); plt.grid(True)
        plt.tight_layout()
        plt.savefig('training_results.png', dpi=150, bbox_inches='tight')
        print("Training plots saved: training_results.png")
    except Exception as e:
        print("Could not save plots:", e)
    
    env.close()
    return agent


# ==================== MAIN ====================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  TRAFFIC & CROWD DETECTION - RL MODEL TRAINING (MODIFIED)")
    print("="*60 + "\n")
    trained_agent = train_rl_model(
        episodes=DEFAULT_EPISODES,
        save_path='models/trained_model_fast.pth'
    )
    print("\n✓ Training run finished.\n")
