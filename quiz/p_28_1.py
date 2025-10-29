import pygame
import sys

# pygame initialization
pygame.init()

# screen , 화면 생성
screen_surface = pygame.display.set_mode((640, 480))

# rendering loop
while True:
    
    screen_surface.fill((0, 0, 0))  # fill the screen with black
    
    # draw rect
    pygame.draw.rect(screen_surface, 
                     (255, 0, 0), # 빨간색 창
                     (100, 100, 100, 50), # 위치는 (100,100), 크기는 100x50
                     2 # 두께
                     )   
     
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
