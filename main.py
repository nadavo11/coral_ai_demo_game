import pygame
import random
import numpy as np

# Initialize Pygame
pygame.init()

# Set up the display
width, height = 1200, 900
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Balloon Bounce")

# Set up colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)

# Set up the balloon
class Baloon():
    def __init__(self):

        self.radius = 40
        self.x = np.array([width / 2,height / 2])
        self.v = np.array([0.1,0])

    def bounce(self,loc):
        dist = np.linalg.norm(self.x - loc)
        if dist < self.radius:
            self.v +=  0.05 * (self.x - loc)
        return 0

    def show(self):
        # Draw the balloon
        pygame.draw.circle(screen, RED, baloon.x, baloon.radius)

    def update(self):
        # Check if balloon hits the ground
        if self.x[1] + self.radius >= height:
            self.v *= [1, -1]
        if self.x[1] < 0:
            self.v *= [1, -0.5]
            self.x[1] += 2

        if self.radius >= self.x[0] or self.x[0] >= width - self.radius:
            self.v *= [-1, 1]

        self.v += g
        self.v *= 0.999
        # Move the balloon
        self.x += self.v



# Game loop
running = True
baloon = Baloon()
g = (0,0.005)

while running:
    screen.fill(BLACK)

    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Get mouse position
    mouse = pygame.mouse.get_pos()

    # Bounce balloon if it hits the mouse
    baloon.bounce(mouse)
    baloon.update()
    baloon.show()

    # Update the display
    pygame.display.flip()


# Quit the game
pygame.quit()
