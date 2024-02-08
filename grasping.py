import cv2
import numpy as np
import math
import time
from datetime import datetime
global graspable
graspable=0

def white_bg(image, focused_object_bbox, name):
    white_bg_img = np.ones_like(image) * 255

    x, y, w, h = focused_object_bbox
    white_bg_img[y:y+h, x:x+w] = image[y:y+h, x:x+w]

    with open('grasp.txt') as file:
        graspableobjects = file.read()
        if name in graspableobjects:
            global graspable
            graspable=1
            screenshot = white_bg_img.copy()
            cv2.imwrite("grasp_this.jpg", screenshot)

    return white_bg_img


vid = cv2.VideoCapture(0)
dimension = 320
confThreshold = 0.5
nmsThreshold = 0.3

classNames = []
with open('coco.names','rt') as f:
    classNames = f.read().rstrip('n').split('\n') 

net = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def Euclid(x1,y1,x2,y2):
    dist = int(math.sqrt((x1-x2)**2+(y1-y2)**2))
    return dist

def findObjects(outputs, img):
    height, width, RGBA = img.shape
    bbox = []
    classIds = []
    confidence_table = []
    centers = []

    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w, h = int(det[2] * width), int(det[3] * height)
                x, y = int((det[0] * width) - w / 2), int((det[1] * height) - h / 2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confidence_table.append(float(confidence))
                centerX, centerY = x + w / 2, y + h / 2
                centers.append((centerX, centerY))

    distances = [Euclid(centerX, centerY, width/2, height/2) for (centerX, centerY) in centers]

    if len(distances) > 0:
        minDistIndex = np.argmin(distances)
    else:
        return

    box = bbox[minDistIndex]
    classId = classIds[minDistIndex]
    name = classNames[classId]
    x, y, w, h = box[0], box[1], box[2], box[3]
    white_bg_img = white_bg(img, box, name)
    cv2.imshow("Screenshot this", white_bg_img)
    global graspable
    if graspable==0 :
        cv2.rectangle(img, (x-7, y-7), (x+w+7, y+h+7), (0,0,255), 2)
        cv2.putText(img, "Not Graspable",
                (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 252), 2)
    else:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0,252,0), 2)
        cv2.putText(img, f'{name.upper()} {int(confidence_table[minDistIndex]*100)}%' " (Graspable)",
                (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        ratio = h/w
        if ratio > 4:
            cv2.circle(img, (x, y+int(h/2)), 10, (255,0,0),cv2.FILLED, 2)
            cv2.circle(img, (x+w, y+int(0.55*h)), 10, (255,0,0),cv2.FILLED, 2)
            cv2.circle(img, (x+w, y+int(0.45*h)), 10, (255,0,0),cv2.FILLED, 2) 
        elif  ratio > 1:
            cv2.circle(img, (x, y+int(h/2)), 10, (255,0,0),cv2.FILLED, 2)
            cv2.circle(img, (x+w, y+int(0.5*h)), 10, (255,0,0),cv2.FILLED, 2)
            cv2.circle(img, (x+w, y+int(0.7*h)), 10, (255,0,0),cv2.FILLED, 2)
            cv2.circle(img, (x+w, y+int(0.6*h)), 10, (255,0,0),cv2.FILLED, 2)
            cv2.circle(img, (x+w, y+int(0.4*h)), 10, (255,0,0),cv2.FILLED, 2)
        elif ratio > 0.25:
            cv2.circle(img, (x+int(w/2), y), 10, (255,0,0),cv2.FILLED, 2)
            cv2.circle(img, (x+int(0.4*w), y+h), 10, (255,0,0),cv2.FILLED, 2)
            cv2.circle(img, (x+int(0.5*w), y+h), 10, (255,0,0),cv2.FILLED, 2)
            cv2.circle(img, (x+int(0.6*w), y+h), 10, (255,0,0),cv2.FILLED, 2)
            cv2.circle(img, (x+int(0.7*w), y+h), 10, (255,0,0),cv2.FILLED, 2)
        else:
            cv2.circle(img, (x+int(w/2), y), 10, (255,0,0),cv2.FILLED, 2)
            cv2.circle(img, (x+int(0.45*w), y+h), 10, (255,0,0),cv2.FILLED, 2)
            cv2.circle(img, (x+int(0.55*w), y+h), 10, (255,0,0),cv2.FILLED, 2)
       
    graspable=0            


start_time = datetime.now()

while True:
    success, img = vid.read()

    blob = cv2.dnn.blobFromImage(img, 1/255,(dimension,dimension),[0,0,0],crop=False)
    net.setInput(blob)

    layerNames = net.getLayerNames()
    outputNames = [layerNames[i-1] for i in net.getUnconnectedOutLayers()]

    outputs = net.forward(outputNames)
    findObjects(outputs,img)

    cv2.imshow('Image', img)
    cv2.waitKey(1)
    runtime = datetime.now() - start_time
    print("Run time: ", runtime)
    # if runtime.total_seconds() >= 60:
    # cv2.destroyAllWindows()
    # bre/a