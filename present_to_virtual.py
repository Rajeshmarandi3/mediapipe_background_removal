import cv2  # python library for computer vision (cv)
import cvzone  # python library for advance computer vision (cv)
from cvzone.SelfiSegmentationModule import SelfiSegmentation  # python library for Image Segmentation
import os  # for operating system (os) command

cap = cv2.VideoCapture(0)  # to open a webcam in python
if cap.isOpened() == False:  # an if condition to check proper webcam access
    print("Error reading video file")

cap.set(cv2.CAP_PROP_FPS, 60)  # set FPS (frame per second)
segmentor = SelfiSegmentation()  # here will are calling SelfiSegmentation to be ready

listImg = os.listdir("images")  # this command lists all the files from 'images' folder
print(listImg)

imgList = []
for imgPath in listImg:  # a for loop to store all the images one by one in a matrix form to imgList
    img = cv2.imread(f"images/{imgPath}")  # this is how we read an image in Python
    width = 640
    height = 480
    dim = (width, height)
    # We have to resize an image to 640x480 for background change to work
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)  # cv2.resize() resizes an image using interpolation
    imgList.append(resized)

indexImg = 0
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # defining the format of video to be stored
out = cv2.VideoWriter('output2.avi', fourcc, 30.0, (640, 480))  # preparing a video file to capture frames

while True:  # a while loop to run the webcam until interruption
    success, img = cap.read()  # take an image frame from a webcam
    if success == True:
        # segmentor.removeBG overlays a person to any given image as a background
        # threshold gives us freedom on how sharp we want background to be removed (value between 0 and 1 )
        img_out = segmentor.removeBG(img, imgList[indexImg], threshold=0.6)
        # cvzone.stackImages is an optinal step where we join original and changed background image side by side
        img_stacked = cvzone.stackImages([img, img_out], 2, 1)

        out.write(img_out)  # this stores each frames to the above video.write file prepared to store frames
        cv2.imshow("Image", img_stacked)  # this opens a window and show our AI processed result
        key = cv2.waitKey(1)  # cv2.waitKey takes command from keybaord to python file
        if key == ord('a'):  # if key 'a' is pressed, we will select the previous image as a background
            if indexImg > 0:
                indexImg = indexImg - 1
            else:
                indexImg = len(imgList) - 1
        elif key == ord('d'):  # if key 'd' is pressed, we will select the previous image as a background
            if indexImg < len(imgList) - 1:
                indexImg = indexImg + 1
            else:
                indexImg = 0
        elif key == ord('q'):  # if key 'q' is pressed, we will stop our code
            break
    else:
        break
# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
