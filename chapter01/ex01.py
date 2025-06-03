import cv2
import pygame
import sys

# pygame initialization
pygame.init()

# screen , 화면 생성
screen_surface = pygame.display.set_mode((640, 480))

# font clock create
font = pygame.font.SysFont(None, 24)
clock = pygame.time.Clock()

# rendering loop
while True:
    
    screen_surface.fill((0, 0, 0))  # fill the screen with black
    
    # draw rect
    pygame.draw.rect(screen_surface, 
                     (0, 255, 0), 
                     (100, 100, 200, 150),
                     2
                     )  
    # draw info text
    text_surface = font.render("Press ESC to exit", True, (255, 255, 255))
    screen_surface.blit(text_surface, (10, 10))
    
    # update the display
    pygame.display.flip()
    
    # event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                pygame.quit()
                sys.exit()

