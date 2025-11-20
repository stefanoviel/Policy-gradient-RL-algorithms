import random
import sys
from typing import List, Tuple

import pygame


CELL_SIZE = 20
GRID_WIDTH = 32
GRID_HEIGHT = 24
SCREEN_WIDTH = CELL_SIZE * GRID_WIDTH
SCREEN_HEIGHT = CELL_SIZE * GRID_HEIGHT
SNAKE_SPEED = 10
MIN_SPEED = 5
MAX_SPEED = 30

BACKGROUND = (30, 30, 30)
SNAKE_COLOR = (0, 200, 0)
FOOD_COLOR = (200, 50, 50)
TEXT_COLOR = (220, 220, 220)

Vector = Tuple[int, int]


def random_food_position(snake: List[Vector]) -> Vector:
    """Return a random grid coordinate not currently occupied by the snake."""
    while True:
        position = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))
        if position not in snake:
            return position


def draw_block(surface: pygame.Surface, color: Tuple[int, int, int], position: Vector) -> None:
    rect = pygame.Rect(position[0] * CELL_SIZE, position[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
    pygame.draw.rect(surface, color, rect)


def main() -> None:
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Snake")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 22)

    def reset_game():
        direction = (1, 0)
        head_x = GRID_WIDTH // 2
        head_y = GRID_HEIGHT // 2
        snake = [(head_x - i, head_y) for i in range(3)]
        food = random_food_position(snake)
        return snake, direction, direction, food, 0

    snake, direction, pending_direction, food, score = reset_game()
    speed = SNAKE_SPEED
    game_over = False

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    pygame.quit()
                    sys.exit()

                key_to_direction = {
                    pygame.K_w: (0, -1),
                    pygame.K_UP: (0, -1),
                    pygame.K_s: (0, 1),
                    pygame.K_DOWN: (0, 1),
                    pygame.K_a: (-1, 0),
                    pygame.K_LEFT: (-1, 0),
                    pygame.K_d: (1, 0),
                    pygame.K_RIGHT: (1, 0),
                }

                if event.key in key_to_direction and not game_over:
                    new_direction = key_to_direction[event.key]
                    if (new_direction[0] != -direction[0]) or (new_direction[1] != -direction[1]):
                        pending_direction = new_direction

                if event.key == pygame.K_r and game_over:
                    snake, direction, pending_direction, food, score = reset_game()
                    speed = SNAKE_SPEED
                    game_over = False

                if event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    speed = min(MAX_SPEED, speed + 1)

                if event.key in (pygame.K_MINUS, pygame.K_UNDERSCORE, pygame.K_KP_MINUS):
                    speed = max(MIN_SPEED, speed - 1)

        if not game_over:
            direction = pending_direction
            new_head = (snake[0][0] + direction[0], snake[0][1] + direction[1])
            snake.insert(0, new_head)

            hit_wall = not (0 <= new_head[0] < GRID_WIDTH and 0 <= new_head[1] < GRID_HEIGHT)
            hit_self = new_head in snake[1:]
            if hit_wall or hit_self:
                game_over = True
            else:
                if new_head == food:
                    score += 1
                    food = random_food_position(snake)
                else:
                    snake.pop()

        screen.fill(BACKGROUND)
        for segment in snake:
            draw_block(screen, SNAKE_COLOR, segment)
        draw_block(screen, FOOD_COLOR, food)

        score_surface = font.render(f"Score: {score}", True, TEXT_COLOR)
        screen.blit(score_surface, (10, 10))

        if game_over:
            text = font.render("Game over! Press R to restart.", True, TEXT_COLOR)
            text_rect = text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
            screen.blit(text, text_rect)

        pygame.display.flip()
        clock.tick(speed)


if __name__ == "__main__":
    main()
