import inline as inline
import matplotlib
import numpy as np

import matplotlib.pyplot as plt

import cv2



# img_raw=cv2.imread(r'C:\Users\Thanos\Downloads\mandrill_colour.png')
# img=cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
# image_blank=np.zeros(shape=(512,512,3),dtype=np.int16)
#
#     #if cv2.waitKey(10) & 0xFF == 27:
#         #break
# plt.imshow(img)
# plt.show()
#
#
# line_red = cv2.line(image_blank, (511,0), (0, 511), (255,0,0), 5)
# plt.imshow(line_red)
# plt.show()
#
#
#
#
#
# #cv2.imwrite('edit1.png',img)
# cv2.destroyAllWindows()
def BGR_to_RGB(img_bgr):
     img=cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
     return img

test_image= cv2.imread(r'C:\Users\Thanos\Downloads\test1.jfif')
#ti_gray= cv2.cvtColor(test_image,cv2.COLOR_BGR2GRAY)
#cv2.imshow('',ti_gray)
#plt.imshow(BGR_to_RGB(ti_gray))

#plt.show()

haar_cascade_face = cv2.CascadeClassifier(r"C:\Users\Thanos\PycharmProjects\Frec\venv\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml")

# face_rec=haar_cascade_face.detectMultiScale(ti_gray, scaleFactor=1.2, minNeighbors=5)
# Num_faces=len(face_rec)
# print(face_rec)
# print(Num_faces)
#
# for (x,y,w,h) in face_rec:
#
#      cv2.rectangle(test_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
# plt.imshow(BGR_to_RGB(test_image))
# plt.show()

def detect_faces(cascade,test_image,scalefactor=1.1):
    img_copy=test_image.copy()
    ti_gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    face_rec = cascade.detectMultiScale(ti_gray, scaleFactor=scalefactor, minNeighbors=5)
    num_faces=len(face_rec)
    for (x, y, w, h) in face_rec:
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return img_copy
def display_detected_faces(cascade,test_image):
    return (BGR_to_RGB(detect_faces(cascade,test_image)))

vid_cap=cv2.VideoCapture(0)
ret,frame= vid_cap.read()
while True:
    cv2.imshow("",detect_faces(haar_cascade_face,frame))
    if cv2.waitKey(1) & 0xFF == 27:
        break
plt.imshow(display_detected_faces(haar_cascade_face,frame))
plt.show()