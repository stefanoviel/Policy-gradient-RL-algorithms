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
FPS = 10

class Snake(gym.Env):
    metadata = {"render_modes": ["human", ], "render_fps": FPS}

    def __init__(self, render_mode=None):
        super(Snake, self).__init__()
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=2, shape=(GRID_SIZE, GRID_SIZE), dtype=np.uint8)
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.snake = [(GRID_SIZE // 2, GRID_SIZE // 2)]
        self.direction = (0, 1)
        self.score = 0
        self._place_food()
        
        if self.render_mode == "human":
            self.render()
            
        return self._get_observation(), {}

    def step(self, action):
        if action == 0: self.direction = (0, -1)
        elif action == 1: self.direction = (0, 1)
        elif action == 2: self.direction = (-1, 0)
        elif action == 3: self.direction = (1, 0)

        new_head = (self.snake[0][0] + self.direction[0], self.snake[0][1] + self.direction[1])

        terminated = False
        if (new_head in self.snake or
            new_head[0] < 0 or new_head[0] >= GRID_SIZE or
            new_head[1] < 0 or new_head[1] >= GRID_SIZE):
            terminated = True
            return self._get_observation(), -10, terminated, False, {}

        self.snake.insert(0, new_head)

        if new_head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()
            reward = -0.1

        if self.render_mode == "human":
            self.render()

        return self._get_observation(), reward, terminated, False, {}


    def render(self):
        if self.render_mode != "human":
            raise NotImplementedError("Render mode not supported: {}".format(self.render_mode))
        
        if self.screen is None:
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("Snake AI")
        if self.clock is None:
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((WIDTH, HEIGHT))
        canvas.fill((0, 0, 0))

        for segment in self.snake:
            pygame.draw.rect(canvas, (0, 255, 0), (segment[0] * TILE_SIZE, segment[1] * TILE_SIZE, TILE_SIZE, TILE_SIZE))
        
        pygame.draw.rect(canvas, (255, 0, 0), (self.food[0] * TILE_SIZE, self.food[1] * TILE_SIZE, TILE_SIZE, TILE_SIZE))

        self.screen.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])


    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()

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

if __name__ == "__main__":
    env = Snake(render_mode="human")
    obs, info = env.reset()
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Obs:\n{obs}\nReward: {reward}\n")
        if terminated or truncated:
            obs, info = env.reset()
    env.close()
