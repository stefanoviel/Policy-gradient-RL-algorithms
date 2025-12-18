import pygame
import random
import sys
import gymnasium as gym
from gymnasium import spaces
import numpy as np


pygame.init()

WIDTH, HEIGHT = 500, 500
GRID_SIZE = 10
TILE_SIZE = WIDTH // GRID_SIZE
FPS = 5


class Snake(gym.Env):

    def __init__(self):
        super(Snake, self).__init__()
        self.action_space = spaces.Discrete(4)  # Up, Down, Left, Right
        self.observation_space = spaces.Box(low=0, high=2, shape=(GRID_SIZE, GRID_SIZE), dtype=np.uint8)
        self.snake = None
        self.direction = None
        self.food = None
        self.score = 0
        self.reset()

    def reset(self):
        self.snake = [(GRID_SIZE // 2, GRID_SIZE // 2)]
        self.direction = (0, 1)
        self.score = 0
        self._place_food()
        return self._get_observation()
    

    def step(self, action):
        if action == 0:   # Up
            self.direction = (0, -1)
        elif action == 1: # Down
            self.direction = (0, 1)
        elif action == 2: # Left
            self.direction = (-1, 0)
        elif action == 3: # Right
            self.direction = (1, 0)

        new_head = (self.snake[0][0] + self.direction[0], self.snake[0][1] + self.direction[1])

        # Check for collisions
        if (new_head in self.snake or
            new_head[0] < 0 or new_head[0] >= GRID_SIZE or
            new_head[1] < 0 or new_head[1] >= GRID_SIZE):
            done = True
            reward = -10
            return self._get_observation(), reward, done, {}

        self.snake.insert(0, new_head)

        if new_head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()
            reward = -0.1

        done = False
        return self._get_observation(), reward, done, {}

    
    def _place_food(self):
        while True:
            self.food = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))
            if self.food not in self.snake:
                break
    
    def _get_observation(self):
        obs = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
        for x, y in self.snake:
            obs[y, x] = 1
        food_x, food_y = self.food
        obs[food_y, food_x] = 2
        return obs
    
