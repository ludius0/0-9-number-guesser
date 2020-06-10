import pygame
import numpy as np

import values as v

### CONSTANTS ###
C = v.C
H = v.H
W = v.W
BLACK = v.BLACK
WHITE = v.WHITE

sz = size_of_image = 28

class Board():
    def __init__(self, screen):
        self.board = np.zeros((1, sz, sz, 1), dtype=int)
        self.screen = screen

    def draw_board(self):
        x, y = 0, 0
        for _ in range(H):
            x, y = x+C, y+C
            pygame.draw.line(self.screen, BLACK, (x,0),(x,W))
            pygame.draw.line(self.screen, BLACK, (0,y),(H,y))

    def draw_pixel(self, x, y, color):
       pygame.draw.rect(self.screen, color, (x, y, C+1, C+1))


    def save_pixel(self, x, y, n):
        x, y = int(x/v.C), int(y/v.C)
        self.board[0][y][x] = n

    def restart_board(self):
        self.board = np.zeros((1, sz, sz, 1), dtype=int)
        for y in range(0, v.H, v.C):
            for x in range(0, v.H, v.C):
                pygame.draw.rect(self.screen, WHITE, (x, y, C+1, C+1))

        
