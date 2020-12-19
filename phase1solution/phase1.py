# Abhinav Chauhan


# Importing Libraries

import cv2
import numpy as np



# Resizing image ( This step is optional)
widthImg = 500
heightImg = 540



# Function to pre-process the image.

def preProcessing(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray,(5,5),1)
    imgCanny = cv2.Canny(imgBlur,200,200)
    kernel = np.ones((5,5))
    imgDial = cv2.dilate(imgCanny,kernel, iterations = 2)
    imgThres = cv2.erode(imgDial, kernel, iterations = 1)

    return imgThres

# Function to get the desired countours and draw corner points around them

def getContours(img):
    biggest = np.array([])
    maxArea = 0
    #_, contours, _= cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area> 5000:
            #cv2.drawContours(imgContour, cnt, -1, (255,0,0), 3)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.03*peri, True)
            if area > maxArea and len(approx) == 4:
                biggest = approx
                maxArea = area

    cv2.drawContours(imgContour, biggest, -1, (255,0,0), 20)
    cv2.imshow('Corner', imgContour)
    return biggest

# This is the main function that resolves the issue posted in the repo. i.e to get corret order of the corner points of the contour detected.

def reorder(myPoints):
    myPoints = myPoints.reshape((4,2))
    myPointsNew = np.zeros((4,1,2), np.int32)
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew

# Function to wrap the image with the ordered points i.e. (0,0) (width, 0) (0, height) and (width, height)

def getWarp(img, biggest):
    biggest = reorder(biggest)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0,0],[widthImg,0],[0,heightImg],[widthImg,heightImg]])
    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    imgOutput = cv2.warpPerspective(img,matrix,(widthImg,heightImg))

    return imgOutput



# code to read the image and display the desired results.
path = "1.jpg"
img = cv2.imread(path)
img = cv2.resize(img, (widthImg, heightImg))
imgContour = img.copy()
imgThres = preProcessing(img)
biggest = getContours(imgThres)

imgWarped = getWarp(img,biggest)
cv2.imshow('Result',imgWarped)
cv2.waitKey(0)
