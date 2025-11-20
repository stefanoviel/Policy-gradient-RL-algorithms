import random
from typing import List, Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - fallback for classic gym installs
    import gym
    from gym import spaces

try:
    import pygame
except ImportError:  # pragma: no cover - pygame is optional for rendering
    pygame = None


Vector = Tuple[int, int]


class SnakeEnv(gym.Env):
    """Gym-style environment for the classic Snake game with a vector state space."""

    metadata = {"render_modes": ["human"], "render_fps": 15}

    RIGHT: Vector = (1, 0)
    DOWN: Vector = (0, 1)
    LEFT: Vector = (-1, 0)
    UP: Vector = (0, -1)
    DIRECTIONS = [RIGHT, DOWN, LEFT, UP]

    def __init__(self, grid_width: int = 20, grid_height: int = 20, render_mode: Optional[str] = None) -> None:
        super().__init__()
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.render_mode = render_mode
        self.cell_size = 20

        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(11,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)  # straight, right turn, left turn

        self.snake: List[Vector] = []
        self.direction: Vector = self.RIGHT
        self.food: Vector = (0, 0)
        self.score = 0
        self.steps_since_food = 0

        self.window = None
        self.clock = None
        self.font = None
        self.render_fps = self.metadata["render_fps"]
        self.display_fps = float(self.render_fps)
        self.button_speed_up = None
        self.button_speed_down = None

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        center_x = self.grid_width // 2
        center_y = self.grid_height // 2
        self.direction = self.RIGHT
        self.snake = [(center_x, center_y), (center_x - 1, center_y), (center_x - 2, center_y)]
        self.food = self._place_food()
        self.score = 0
        self.steps_since_food = 0

        observation = self._get_state()
        info = {"score": self.score, "length": len(self.snake)}
        return observation, info

    def step(self, action: int):
        action = int(action)
        assert self.action_space.contains(action)

        self.direction = self._direction_after_action(action)
        new_head = (self.snake[0][0] + self.direction[0], self.snake[0][1] + self.direction[1])
        
        current_dist = np.linalg.norm(np.array(self.snake[0]) - np.array(self.food))
        new_dist = np.linalg.norm(np.array(new_head) - np.array(self.food))

        reward = 0.0
        terminated = False

        if self._is_collision(new_head):
            reward = -10.0
            terminated = True
        else:
            self.snake.insert(0, new_head)
            
            if new_dist < current_dist:
                reward += 0.1
            else:
                reward -= 0.1

            if new_head == self.food:
                reward = 10.0
                self.score += 1
                self.food = self._place_food()
                self.steps_since_food = 0
            else:
                self.snake.pop()
                self.steps_since_food += 1
                
        reward -= 0.01

        observation = self._get_state()
        info = {"score": self.score, "length": len(self.snake)}
        truncated = False

        return observation, reward, terminated, truncated, info
    def render(self):
        if self.render_mode != "human":
            return
        if pygame is None:
            raise RuntimeError("pygame is required for rendering but is not installed.")

        if self.window is None:
            pygame.init()
            width = self.grid_width * self.cell_size
            height = self.grid_height * self.cell_size
            self.window = pygame.display.set_mode((width, height))
            pygame.display.set_caption("Snake RL Environment")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("Arial", 18)
            self.render_fps = self.metadata["render_fps"]
            self.display_fps = float(self.render_fps)
            padding = 10
            btn_w, btn_h = 70, 30
            self.button_speed_up = pygame.Rect(padding, padding, btn_w, btn_h)
            self.button_speed_down = pygame.Rect(padding + btn_w + 10, padding, btn_w, btn_h)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                raise SystemExit
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    self._increase_speed()
                if event.key in (pygame.K_MINUS, pygame.K_UNDERSCORE, pygame.K_KP_MINUS):
                    self._decrease_speed()
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self.button_speed_up and self.button_speed_up.collidepoint(event.pos):
                    self._increase_speed()
                if self.button_speed_down and self.button_speed_down.collidepoint(event.pos):
                    self._decrease_speed()

        self.window.fill((30, 30, 30))

        for segment in self.snake:
            self._draw_block(segment, (0, 200, 0))
        self._draw_block(self.food, (200, 50, 50))
        self._draw_ui()

        pygame.display.flip()
        delta_ms = self.clock.tick(self.render_fps)
        if delta_ms > 0:
            self.display_fps = 1000.0 / delta_ms

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None

    def _direction_after_action(self, action: int) -> Vector:
        idx = self.DIRECTIONS.index(self.direction)
        if action == 1:  # right turn
            idx = (idx + 1) % 4
        elif action == 2:  # left turn
            idx = (idx - 1) % 4
        return self.DIRECTIONS[idx]

    def _is_collision(self, point: Vector) -> bool:
        x, y = point
        out_of_bounds = x < 0 or x >= self.grid_width or y < 0 or y >= self.grid_height
        hits_body = point in self.snake
        return out_of_bounds or hits_body

    def _place_food(self) -> Vector:
        available = [(x, y) for x in range(self.grid_width) for y in range(self.grid_height) if (x, y) not in self.snake]
        if not available:
            raise RuntimeError("No space left to place food.")
        return random.choice(available)

    def _get_state(self) -> np.ndarray:
        head_x, head_y = self.snake[0]
        dir_idx = self.DIRECTIONS.index(self.direction)
        dir_left = self.DIRECTIONS[(dir_idx - 1) % 4]
        dir_right = self.DIRECTIONS[(dir_idx + 1) % 4]

        danger_straight = self._is_collision((head_x + self.direction[0], head_y + self.direction[1]))
        danger_right = self._is_collision((head_x + dir_right[0], head_y + dir_right[1]))
        danger_left = self._is_collision((head_x + dir_left[0], head_y + dir_left[1]))

        state = np.array(
            [
                float(danger_straight),
                float(danger_right),
                float(danger_left),
                float(self.direction == self.LEFT),
                float(self.direction == self.RIGHT),
                float(self.direction == self.UP),
                float(self.direction == self.DOWN),
                float(self.food[0] < head_x),
                float(self.food[0] > head_x),
                float(self.food[1] < head_y),
                float(self.food[1] > head_y),
            ],
            dtype=np.float32,
        )
        return state

    def _draw_block(self, position: Vector, color: Tuple[int, int, int]) -> None:
        if self.window is None or pygame is None:
            return
        rect = pygame.Rect(position[0] * self.cell_size, position[1] * self.cell_size, self.cell_size, self.cell_size)
        pygame.draw.rect(self.window, color, rect)

    def _draw_ui(self) -> None:
        if self.window is None or self.font is None or pygame is None:
            return
        button_color = (60, 60, 60)
        text_color = (230, 230, 230)
        highlight_color = (100, 100, 100)

        if self.button_speed_up:
            pygame.draw.rect(self.window, button_color, self.button_speed_up)
            pygame.draw.rect(self.window, highlight_color, self.button_speed_up, 2)
            plus_text = self.font.render("+ Speed", True, text_color)
            text_rect = plus_text.get_rect(center=self.button_speed_up.center)
            self.window.blit(plus_text, text_rect)

        if self.button_speed_down:
            pygame.draw.rect(self.window, button_color, self.button_speed_down)
            pygame.draw.rect(self.window, highlight_color, self.button_speed_down, 2)
            minus_text = self.font.render("- Speed", True, text_color)
            text_rect = minus_text.get_rect(center=self.button_speed_down.center)
            self.window.blit(minus_text, text_rect)

        info_text = self.font.render(f"Target: {self.render_fps} FPS  Actual: {self.display_fps:.1f} FPS", True, text_color)
        self.window.blit(info_text, (10, 50))

    def _increase_speed(self) -> None:
        self.render_fps = min(self.render_fps * 2, 240)

    def _decrease_speed(self) -> None:
        self.render_fps = max(self.render_fps // 2, 5)


if __name__ == "__main__":
    env = SnakeEnv(render_mode="human")
    obs, info = env.reset()
    print("Starting demo. Close the window to stop.")
    try:
        while True:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            if terminated or truncated:
                obs, info = env.reset()
    except SystemExit:
        pass
    finally:
        env.close()
