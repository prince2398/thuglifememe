import cv2
from PIL import Image
import sys

face_cascade_file = "haarcascade_frontalface_default.xml"
image_path = sys.argv[1]
mask_path = "mask.png"

face_cascade = cv2.CascadeClassifier(face_cascade_file)


img = cv2.imread(image_path , 1)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
image = Image.open(image_path)

faces = face_cascade.detectMultiScale(gray,1.15)


for (x,y,w,h) in faces:
	box = (x,y)
	mask = Image.open(mask_path)
	mask = mask.resize((w,h))
	image.paste(mask, box , mask = mask)
	
image.save("thug.png")