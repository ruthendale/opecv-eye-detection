import cv2 as c
import numpy as np
frameWidth = 640
frameHeight = 480
v = c.VideoCapture(0)
v.set(3, frameWidth)
v.set(4, frameHeight)
v.set(10,150)
#i = c.imread("people.jpg")
cpath = "LearnOpenCV/haarcascades/haarcascade_eye.xml"
ca = c.CascadeClassifier(cpath)
while(1):
    _,i = v.read()
    g = c.cvtColor(i,c.COLOR_BGR2GRAY)
    fs = ca.detectMultiScale(
    g,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30,30),
    flags=c.CASCADE_SCALE_IMAGE,
    )
    for (o,u,l,b) in fs:
        c.rectangle(i, (o,u), (o+l, u+b), (0,0,255),2)
    print("Found faces {}".format(len(fs)))
    c.imshow("Title", i)
    k = c.waitKey(10) & 0xff
    if k == 27:
        break