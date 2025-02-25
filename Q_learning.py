# Q_learning.py
import numpy as np
import random

class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995):
        self.alpha = alpha  # 学习率
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # 探索概率
        self.epsilon_decay = epsilon_decay  # 探索概率衰减
        self.q_table = {}  # Q 表

    def get_state(self, snake_head, food_pos, snake_body, width, height):
        # 定义状态：蛇头位置、食物位置、周围障碍物
        head_x, head_y = snake_head
        food_x, food_y = food_pos
        danger_up = head_y - 10 < 0 or [head_x, head_y - 10] in snake_body
        danger_down = head_y + 10 >= height or [head_x, head_y + 10] in snake_body
        danger_left = head_x - 10 < 0 or [head_x - 10, head_y] in snake_body
        danger_right = head_x + 10 >= width or [head_x + 10, head_y] in snake_body

        state = (
            danger_up, danger_down, danger_left, danger_right,
            head_x < food_x, head_x > food_x,
            head_y < food_y, head_y > food_y
        )
        return state

    def choose_action(self, state):
        # ε-greedy 策略选择动作
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(['UP', 'DOWN', 'LEFT', 'RIGHT'])
        else:
            return max(self.q_table.get(state, {'UP': 0, 'DOWN': 0, 'LEFT': 0, 'RIGHT': 0}).items(), key=lambda x: x[1])[0]

    def update_q_table(self, state, action, reward, next_state):
        # 更新 Q 表
        if state not in self.q_table:
            self.q_table[state] = {'UP': 0, 'DOWN': 0, 'LEFT': 0, 'RIGHT': 0}
        if next_state not in self.q_table:
            self.q_table[next_state] = {'UP': 0, 'DOWN': 0, 'LEFT': 0, 'RIGHT': 0}

        old_value = self.q_table[state][action]
        next_max = max(self.q_table[next_state].values())
        new_value = old_value + self.alpha * (reward + self.gamma * next_max - old_value)
        self.q_table[state][action] = new_value

        # 衰减探索概率
        self.epsilon *= self.epsilon_decay