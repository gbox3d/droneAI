import cv2
import pygame

import sys

from ultralytics import YOLO,checks

checks()

width, height = 640, 480

# Initialize pygame
pygame.init()
screen_surface = pygame.display.set_mode((width, height))

clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 24)

# Load the YOLO model
model = YOLO("yolo11n.pt")

# Open the webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)


if not cap.isOpened():
    print("Error: Could not open webcam.")
    sys.exit(1)
print("Webcam opened successfully.")

bLoop = True
while bLoop:
    
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from webcam.")
        break    
    
    results = model(frame, conf=0.7,verbose=False)
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
            conf = box.conf[0] * 100  # convert to percentage

            if model.names[cls] == "book" and conf >= 70:
                print(f"Detected a book with confidence {conf}%")
    
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            bLoop = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                bLoop = False





