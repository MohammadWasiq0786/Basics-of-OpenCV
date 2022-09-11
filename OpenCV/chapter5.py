import cv2
import time

cap = cv2.VideoCapture(0)

while True:

     _,img = cap.read()

     img = cv2.flip(img, 1)


     cv2.imshow("VideoStream",img)
     key = cv2.waitKey(1)

cap.release()

# cTime = time.time()
# fps = 1 / (cTime - pTime)
# pTime = cTime
#
# cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
#             (255, 0, 255), 3)

