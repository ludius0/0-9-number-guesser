"""
Programmed by @ludius0
"""
# Modules
import pygame
import numpy as np
import tkinter
from tkinter import messagebox

# Scripts
import activate_model
import display_
import values as v


def show_answer(answer):
    root = tkinter.Tk()
    root.withdraw()
    message = messagebox.showinfo(title="GUESS",message=f"GUESSED NUMBER IS: {answer}")
    root.destroy()
    

### GET POS ###
def get_pos():
    x, y = pygame.mouse.get_pos()
    x, y = round_pos(x, y)
    return (x, y)

def round_pos(x, y):
    a = int(v.C / 2)
    while True:
        if x % v.C != 0:
            if (round(x, -1)-x) >= a:
                x += 1
            else:
                x -= 1
        if y % v.C != 0:
            if (round(y, -1)-y) >= a:
                y += 1
            else:
                y -= 1
        if (x%v.C, y%v.C) == (0, 0):
            break
    return (x, y)

### GUI ###
pygame.init()
screen = pygame.display.set_mode((v.H, v.W))
pygame.display.set_caption("0-9 number guesser by ludius0")
screen.fill(v.WHITE)

img = display_.Board(screen)

while True:
    for event in pygame.event.get():
        pressed_key = pygame.key.get_pressed()
        pressed_mouse = pygame.mouse.get_pressed()

        if event.type == pygame.QUIT:   # Quit
            pygame.quit()

        if pressed_mouse[0]:    # Draw / left click
            x, y = get_pos()
            n = 255             # Pixel color (inversed)

            img.save_pixel(x, y, n)
            img.draw_pixel(x, y, v.BLACK)

        if pressed_mouse[2]:    # Delete / right click
            x, y = get_pos()
            n = 0

            img.save_pixel(x, y, n)
            img.draw_pixel(x, y, v.WHITE)
        
        if pressed_key[pygame.K_SPACE]: # Use deep_learning model
            if img.board.any() == True:
                answer = activate_model.guesser(img.board)
                show_answer(answer)
                img.restart_board()
            
            
    img.draw_board()
    pygame.display.update()




