import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import random 
import argparse
import cv2
import time



if __name__ == '__main__':

    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

    cap = cv2.VideoCapture(0)

    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    assert cap.isOpened(), 'Cannot capture source'
    
    frames = 0
    start = time.time()    
    while cap.isOpened():

        ret,frame = cap.read()
        if ret:
            img=frame
            #img = cv2.resize(frame, None, fx=0.8, fy=0.8)
            height, width, channels = img.shape

            blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            outs = net.forward(output_layers)

            class_ids = []
            confidences = []
            boxes = []

            for out in outs:
                #print(out.shape,outs.shape)
                for detection in out:
                 #   print(detection)
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        # Object detected
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        # Rectangle coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        boxes.append([x, y, w, h])

                        confidences.append(float(confidence))
                        class_ids.append(class_id)
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            
            font = cv2.FONT_HERSHEY_PLAIN
            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = "{}: {:.2f}%".format(str(classes[class_ids[i]]), confidences[i] * 100)
                    color = colors[i]
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(img, label, (x, y + 30), font, 3, color, 3)
            cv2.imshow("Frame", img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
        else:
            break
        frames += 1

    cv2.destroyAllWindows()
