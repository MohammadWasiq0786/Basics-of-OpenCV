import cv2
import numpy as np

def stackImages(imgArray,scale,lables=[]):
    sizeW= imgArray[0][0].shape[1]
    sizeH = imgArray[0][0].shape[0]
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                imgArray[x][y] = cv2.resize(imgArray[x][y], (sizeW, sizeH), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
            hor_con[x] = np.concatenate(imgArray[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for x in range(0, rows):
            imgArray[x] = cv2.resize(imgArray[x], (sizeW, sizeH), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        hor_con= np.concatenate(imgArray)
        ver = hor
    if len(lables) != 0:
        eachImgWidth= int(ver.shape[1] / cols)
        eachImgHeight = int(ver.shape[0] / rows)
        print(eachImgHeight)
        for d in range(0, rows):
            for c in range (0,cols):
                cv2.rectangle(ver,(c*eachImgWidth,eachImgHeight*d),(c*eachImgWidth+len(lables[d][c])*13+27,30+eachImgHeight*d),(255,255,255),cv2.FILLED)
                cv2.putText(ver,lables[d][c],(eachImgWidth*c+10,eachImgHeight*d+20),cv2.FONT_HERSHEY_COMPLEX,0.7,(255,0,255),2)
    return ver

def getContours(img):
    contours,heirarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area>70:
            peri = cv2.arcLength(cnt,True)
            print (area,peri)
            cv2.drawContours(imgContour,cnt,-1,(255,0,255),2)
            approx = cv2.approxPolyDP(cnt,0.02*peri,True)
            print(len(approx))
            objcor = len(approx)
            x,y,w,h = cv2.boundingRect(approx)
            cv2.rectangle(imgContour,(x,y),(x+w,y+h),(0,255,255),2)

            if objcor == 3:
                objType ="Triangle"
            elif objcor == 4:
                aspRatio = w/float(h)
                if aspRatio>0.95 and aspRatio<1.05:
                    objType="Square"
                else:objType="Rectangle"

            elif objcor == 5:objType="Pentagon"
            elif objcor == 6: objType="Hexagon"
            elif objcor>6:objType="Curve"
            else : objType= "None"

            cv2.putText(imgContour,objType,(x+(w//2)-35,y+(h//2)+5),cv2.FONT_ITALIC,0.4,(0,0,0),1)


path = 'Resources/Wasiq.jpg'
imgd = cv2.imread(path)
img = cv2.resize(imgd,(600,250))

imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray,(3,3),2)
imgCanny = cv2.Canny(imgBlur,100,100)
imgContour = img.copy()
getContours(imgCanny)
imgStack = stackImages(([img,imgGray,imgGray],[imgBlur,imgCanny,imgContour]),0.6)


cv2.imshow("Shapes",imgContour)
#cv2.imshow("Image Stack",imgStack)
#cv2.imshow("Image Gray",imgGray)
#cv2.imshow("Image Blur",imgBlur)
#cv2.imshow("Image Canny",imgCanny)

cv2.waitKey(0)