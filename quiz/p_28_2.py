import cv2
import pygame
import sys

# pygame initialization
pygame.init()

width, height = 640, 480

# screen , 화면 생성
screen_surface = pygame.display.set_mode((width, height))

# font clock create
font = pygame.font.SysFont(None, 24)
clock = pygame.time.Clock()

cap = cv2.VideoCapture(0)  # Open the default camera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)  # Set width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)  # Set height

if not cap.isOpened():
    print("Error: Could not open camera.")
    sys.exit()

# rendering loop
while True:
    
    #screen_surface.fill((0, 0, 0))  # fill the screen with black
    ret, frame = cap.read()  # Read a frame from the camera
    if not ret:
        print("Error: Could not read frame.")
        break
    
    # Convert the frame to RGB format
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Convert the frame to a pygame surface
    frame_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
    # Blit the frame surface onto the screen
    screen_surface.blit(frame_surface, (0, 0))
    
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

