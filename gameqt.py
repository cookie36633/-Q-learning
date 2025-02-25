import pygame
import random
import heapq
from Q_learning import QLearningAgent

# Initialize Pygame
pygame.init()

# Set game window size
width, height = 640, 480
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Q-learning 贪吃蛇游戏")

# define color
white = (255, 255, 255)
black = (0, 0, 0)
red = (255, 0, 0)
green = (0, 255, 0)

# Parameters of snakes
snake_block_size = 10
snake_speed = 10

# Initialize Q-learning agent
agent = QLearningAgent()

# Heuristic function of A * algorithm
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# Implementation of A * Algorithm
def a_star_search(start, goal, obstacles):
    open_list = []
    heapq.heappush(open_list, (0, start))
    came_from = {}
    cost_so_far = {start: 0}

    while open_list:
        _, current = heapq.heappop(open_list)

        if current == goal:
            break

        for next in [(current[0] + snake_block_size, current[1]),
                     (current[0] - snake_block_size, current[1]),
                     (current[0], current[1] + snake_block_size),
                     (current[0], current[1] - snake_block_size)]:
            if 0 <= next[0] < width and 0 <= next[1] < height and next not in obstacles:
                new_cost = cost_so_far[current] + snake_block_size
                if next not in cost_so_far or new_cost < cost_so_far[next]:
                    cost_so_far[next] = new_cost
                    priority = new_cost + heuristic(goal, next)
                    heapq.heappush(open_list, (priority, next))
                    came_from[next] = current

    # Rebuilding the Path
    if goal not in came_from:
        return []  # If unable to reach the target, return an empty path

    current = goal
    path = []
    while current != start:
        path.append(current)
        current = came_from[current]
    path.reverse()
    return path

# Game main loop
clock = pygame.time.Clock()
episodes = 100  # Training epochs
max_score = 0  # Record the highest score

for episode in range(episodes):
    x1, y1 = width // 2, height // 2
    snake_list = []
    snake_length = 1
    x1_change, y1_change = 0, 0
    score = 0
    game_over = False

    # Generate multiple foods
    num_foods = 5
    foods = []
    for _ in range(num_foods):
        foodx = round(random.randrange(0, width - snake_block_size) / 10.0) * 10.0
        foody = round(random.randrange(0, height - snake_block_size) / 10.0) * 10.0
        foods.append((foodx, foody))

    while not game_over:
        # Get current status
        state = agent.get_state(snake_head=[x1, y1], food_pos=foods[0], snake_body=snake_list, width=width, height=height)

        # Use the A * algorithm to find the path
        obstacles = snake_list[1:]
        path = a_star_search((x1, y1), foods[0], obstacles)

        # If there is a path, select the first action on the path
        if path and len(path) > 1:
            next_step = path[1]
            if next_step[0] > x1:
                action = 'RIGHT'
            elif next_step[0] < x1:
                action = 'LEFT'
            elif next_step[1] > y1:
                action = 'DOWN'
            elif next_step[1] < y1:
                action = 'UP'
        else:
            action = agent.choose_action(state)

        # Update snake head position based on actions
        if action == 'UP':
            x1_change = 0
            y1_change = -snake_block_size
        elif action == 'DOWN':
            x1_change = 0
            y1_change = snake_block_size
        elif action == 'LEFT':
            x1_change = -snake_block_size
            y1_change = 0
        elif action == 'RIGHT':
            x1_change = snake_block_size
            y1_change = 0

        # Update snake head location
        x1 += x1_change
        y1 += y1_change
        snake_head = [x1, y1]
        snake_list.append(snake_head)

        # Check if it hits a wall or oneself (end of turn condition)
        if x1 >= width or x1 < 0 or y1 >= height or y1 < 0 or snake_head in snake_list[:-1]:
            reward = -10
            game_over = True
        else:
            # Check if you have eaten food
            for food in foods[:]:
                if x1 == food[0] and y1 == food[1]:
                    reward = 10  # Eating food and receiving positive rewards
                    foods.remove(food)
                    score += 1
                    snake_length += 1
                    # If food is eaten, regenerate a new food
                    if len(foods) < num_foods:
                        foodx = round(random.randrange(0, width - snake_block_size) / 10.0) * 10.0
                        foody = round(random.randrange(0, height - snake_block_size) / 10.0) * 10.0
                        foods.append((foodx, foody))
                    break
            else:
                reward = -0.1  # Give small negative rewards for each step to encourage quick finding of food
                if len(snake_list) > snake_length:
                    del snake_list[0]

        # Get the next state
        next_state = agent.get_state(snake_head=[x1, y1], food_pos=foods[0], snake_body=snake_list, width=width, height=height)

        # Update Q table
        agent.update_q_table(state, action, reward, next_state)

        # Draw game interface
        screen.fill(black)
        for segment in snake_list:
            pygame.draw.rect(screen, white, [segment[0], segment[1], snake_block_size, snake_block_size])
        for food in foods:
            pygame.draw.rect(screen, green, [food[0], food[1], snake_block_size, snake_block_size])

        # Show score
        font_style = pygame.font.SysFont(None, 30)
        score_text = font_style.render(f"回合: {episode + 1}, 当前分数: {score}, 最高分数: {max_score}", True, white)
        screen.blit(score_text, [10, 10])

        pygame.display.update()
        clock.tick(snake_speed)

        # If the game ends, exit the current round
        if game_over:
            if score > max_score:
                max_score = score
            break

    # 每回合结束后，探索概率递减
    agent.epsilon *= 0.995

    # Clear the game window
    screen.fill(black)
    pygame.display.update()

pygame.quit()