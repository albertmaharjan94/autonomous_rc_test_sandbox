import cv2
import time
import serial
import pygame
pygame.init()
pygame.display.set_mode()
pygame.key.set_repeat()
ser = serial.Serial('/dev/ttyACM0', 19200, timeout=1)
ser.flush()	
SPEED = 0
DIRECTION= 30
def writeArduiono(d, s):
    ACTION = (str(d)+"#" +str(s)+ "\n").encode('utf_8')
    print(ACTION)
    ser.write(ACTION)
    line = ser.readline().decode('utf-8').rstrip()	
    print(line)
def moveLeft():
    DIRECTION = 0
def moveRight():
    DIRECTION = 60
#  main
while True: 
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit(); #sys.exit() if sys is imported
        if event.type == pygame.KEYDOWN:
            if event.key == ord('w'):
                SPEED = 50
            elif event.key == ord('s'):
                SPEED = -50
            if event.key == ord('a'):
                DIRECTION = 60
            elif event.key == ord('d'):
                DIRECTION = 0
        if event.type == pygame.KEYUP:
            if event.key == ord('w') or event.key == ord('s'):
                SPEED = 0
            if event.key == ord('a') or event.key == ord('d'):
                DIRECTION = 30
        # print(DIRECTION, SPEED)
    writeArduiono(DIRECTION, SPEED)
    # DIRECTION = 0
    # writeArduiono(DIRECTION, SPEED)
    # time.sleep(0.2)
    # DIRECTION = 30
    # writeArduiono(DIRECTION, SPEED)
    # time.sleep(0.2)
    # DIRECTION = 60
    # writeArduiono(DIRECTION, SPEED)
    # time.sleep(0.5)