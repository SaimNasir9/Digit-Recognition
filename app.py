import pygame, sys
from pygame.locals import *
from pygame import image
import numpy as np
from numpy import testing
from keras.models import load_model
import cv2

WINDOWSIZEX = 640
WINDOWSIZEY = 480

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

IMAGESAVE = False

MODEL = load_model("mnist_cnn_model.h5")

LABELS = {
    0: "Zero", 1: "One",
    2: "Two", 3: "Three",
    4: "Four", 5: "Five",
    6: "Six", 7: "Seven",
    8: "Eight", 9: "Nine"
}

# Initialization of pygame
pygame.init()

# Define the font AFTER pygame.init()
FONT = pygame.font.Font(None, 18)

# Display size
DISPLAYSURFACE = pygame.display.set_mode((WINDOWSIZEX, WINDOWSIZEY))

# Display caption
pygame.display.set_caption("Digit Recognizer")

# For the screen display
open = True

iswriting = False

number_xcord = []
number_ycord = []

image_count = 0

BOUNDING_BOX_BOUNDRY = 5  # 5 pixels boundary for the boxes

PREDICT = True

while open:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

        if event.type == KEYDOWN:  
            if event.unicode == "n":  
                DISPLAYSURFACE.fill(BLACK) 
        
        if event.type == MOUSEMOTION and iswriting:
            xcord, ycord = event.pos
            pygame.draw.circle(DISPLAYSURFACE, WHITE, (xcord, ycord), 4, 0)

            number_xcord.append(xcord)
            number_ycord.append(ycord)

        if event.type == MOUSEBUTTONDOWN:
            iswriting = True

        if event.type == MOUSEBUTTONUP:
            iswriting = False

            # Ensure coordinates are not empty before processing
            if number_xcord and number_ycord:
                number_xcord = sorted(number_xcord)
                number_ycord = sorted(number_ycord)

                bounding_box_min_x = max(number_xcord[0] - BOUNDING_BOX_BOUNDRY, 0)
                bounding_box_max_x = min(WINDOWSIZEX, number_xcord[-1] + BOUNDING_BOX_BOUNDRY)
                bounding_box_min_y = max(number_ycord[0] - BOUNDING_BOX_BOUNDRY, 0)
                bounding_box_max_y = min(WINDOWSIZEY, number_ycord[-1] + BOUNDING_BOX_BOUNDRY)

                # Draw bounding box around the detected area
                pygame.draw.rect(
                    DISPLAYSURFACE, RED,
                    pygame.Rect(bounding_box_min_x, bounding_box_min_y,
                                bounding_box_max_x - bounding_box_min_x,
                                bounding_box_max_y - bounding_box_min_y), 2
                )

                number_xcord = []
                number_ycord = []

                img_arr = np.array(pygame.PixelArray(DISPLAYSURFACE))[bounding_box_min_x:bounding_box_max_x, bounding_box_min_y:bounding_box_max_y].T.astype(np.float32)
                if IMAGESAVE:
                    cv2.imwrite(f"image_{image_count}.png", img_arr)
                    image_count += 1

                if PREDICT:
                    image = cv2.resize(img_arr, (28, 28))
                    image = np.pad(image, (10, 10), "constant", constant_values=0)
                    image = cv2.resize(image, (28, 28)) / 255

                    label = str(LABELS[np.argmax(MODEL.predict(image.reshape(1, 28, 28, 1)))])
                    textsurface = FONT.render(label, True, RED, WHITE)
                    textObject = textsurface.get_rect()
                    textObject.left, textObject.bottom = bounding_box_min_x, bounding_box_max_y

                    DISPLAYSURFACE.blit(textsurface, textObject)

            # Reset the display if the "n" key is pressed
            if event.type == KEYDOWN:
                if event.unicode == "n":
                    DISPLAYSURFACE.fill(BLACK)

        pygame.display.update()
