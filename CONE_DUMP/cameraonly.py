import cv2
import os

# Read video from disk and count frames
cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    t = cv2.waitKey(20)
    if t == ord('q'):
        break
    cv2.imshow("f",frame)


cap.release()
cv2.destroyAllWindows()
