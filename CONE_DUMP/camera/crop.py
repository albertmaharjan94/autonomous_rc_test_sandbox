import cv2
img = cv2.imread("green.jpg")
crop_img = img[10:30, 10:20]
cv2.imshow("cropped", crop_img)
cv2.waitKey(0)
