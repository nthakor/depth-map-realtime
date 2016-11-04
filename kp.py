import cv2
import numpy as np

img=cv2.imread('lena.jpg')
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
# kp = sift.detect(gray,None)
kp, des = sift.detectAndCompute(gray,None)
print len(des)
img=cv2.drawKeypoints(gray,kp,img)
cv2.imshow('kp',img)
cv2.waitKey(0)