import random
from collections import defaultdict

class QTableLearner:
    def __init__(self, actions, alpha = 0.1, gamma = 0.9, epsilon = 0.2):
        self.q_table = defaultdict(lambda: {action: 0.0 for action in actions })
        self.alpha = alpha # learning rate
        self.gamma = gamma # discount factor
        self.epsilon = epsilon # exploration rate
        self.actions = actions

    def get_state_key(self, hunger, fatigue):
        # Discretize state for simplicity
        h = int(hunger * 10)
        f = int(fatigue * 10)
        return f"h{h}_f{f}"

    def choose_action(self, hunger, fatigue):
        state = self.get_state_key(hunger, fatigue)
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            return max(self.q_table[state], key=self.q_table[state].get)

    def update(self, prev_hunger, prev_fatigue, action, reward, next_hunger, next_fatigue):
        prev_state = self.get_state_key(prev_hunger, prev_fatigue)
        next_state = self.get_state_key(next_hunger, next_fatigue)

        max_future_q = max(self.q_table[next_state].values())
        old_q = self.q_table[prev_state][action]

        new_q = old_q + self.alpha * (reward + self.gamma * max_future_q - old_q)
        self.q_table[prev_state][action] = new_q

    def __str__(self):
        return f"Q-table (sample): {dict(list(self.q_table.items())[:5])}"
