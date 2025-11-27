# rl_agent.py
import numpy as np
import json
import os

class QLearningAgent:
    def __init__(self, max_capacity=10, n_actions=3, alpha=0.1, gamma=0.95, epsilon=0.2, epsilon_decay=0.9995):
        """
        States: discrete counts 0..max_capacity (counts > max_capacity are clipped)
        Actions: 0=No-op, 1=Warning, 2=Emergency
        """
        self.max_capacity = max_capacity
        self.n_states = max_capacity + 1
        self.n_actions = n_actions
        self.Q = np.zeros((self.n_states, n_actions), dtype=float)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = 0.01

    def state_index(self, count):
        c = int(min(count, self.max_capacity))
        return c

    def choose_action(self, count):
        s = self.state_index(count)
        # epsilon-greedy
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.Q[s]))

    def learn(self, s_count, action, reward, next_count):
        s = self.state_index(s_count)
        ns = self.state_index(next_count)
        old = self.Q[s, action]
        target = reward + self.gamma * np.max(self.Q[ns])
        self.Q[s, action] = old + self.alpha * (target - old)
        # decay epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def save(self, path="q_table.json"):
        data = {
            "max_capacity": self.max_capacity,
            "Q": self.Q.tolist(),
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon
        }
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path="q_table.json"):
        if not os.path.exists(path):
            return False
        with open(path, "r") as f:
            data = json.load(f)
        self.max_capacity = data["max_capacity"]
        self.Q = np.array(data["Q"])
        self.n_states = self.Q.shape[0]
        return True

