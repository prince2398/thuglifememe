import numpy as np 
import cv2
from PIL import Image
import pprint

haarcascade_path = "//home//prince//opencv-tmp//opencv-3//data//haarcascades//"
face_cascade_file = "haarcascade_frontalface_default.xml"
eye_cascade_file = "haarcascade_eye.xml"
image_path = input("Enter path to image or only the file name if fileis in same directory : ")


face_cascade = cv2.CascadeClassifier(haarcascade_path + face_cascade_file)
eye_cascade = cv2.CascadeClassifier(haarcascade_path + eye_cascade_file)


img = cv2.imread(image_path , 1)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# 	pass
# else :
# 	print("Error 404 : No Image Found ")
# 	exit()


faces = face_cascade.detectMultiScale(gray,1.3,5)

for (x,y,w,h) in faces:
	roi_gray = gray[y:y+h , x:x+w]
	eyes = eye_cascade.detectMultiScale(roi_gray)
	print(eyes)

print(faces)


