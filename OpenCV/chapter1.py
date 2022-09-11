import cv2
import numpy as np

imgd = cv2.imread("Resources/Lemon.jpg")
# imgd = cv2.imread("Resources/Wasiq.jpg")

kernel = np.ones((5,5),np.uint8)

img = cv2.resize(imgd,(300,300))
imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgCanny = cv2.Canny(img,100,100)
imgDilation = cv2.dilate(imgCanny,kernel,iterations=1)
imgEroded = cv2.erode(imgDilation,kernel,iterations=1)
imgCropped = img[200:400,300:500]
imgHor = np.hstack((imgGray,imgCanny))
imgHor1 = np.hstack((imgDilation,imgEroded))
imgVer = np.vstack((imgHor,imgHor1))

cv2.imshow("Original Image", imgd)
cv2.imshow("Gray Image", imgGray)
# cv2.imshow("Canny Image",imgCanny)
# cv2.imshow("Dilated Image",imgDilation)
# cv2.imshow("Eroded Image",imgEroded)
# cv2.imshow("Image Resize",img)
# cv2.imshow("Image Cropped",imgCropped)
cv2.imshow("Stacked Image",imgVer)

print(img.shape)
cv2.waitKey(0)
