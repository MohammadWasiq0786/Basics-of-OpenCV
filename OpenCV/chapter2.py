import cv2
import numpy as np


img = np.zeros((512,512,3),np.uint8)
img[199:301,199:301] = 0,120,255
cv2.line(img,(200,200),(300,300),(125,0,0),2)
cv2.line(img,(200,300),(300,200),(125,0,0),2)
cv2.rectangle(img,(199,199),(301,301),(200,0,135),2)
cv2.circle(img,(250,250),100,(255,255,0),2)
cv2.putText(img,"Project",(200,100),cv2.FONT_ITALIC,1,(120,20,120),2)


cv2.imshow("Output",img)
cv2.waitKey(0)
