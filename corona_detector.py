import cv2
import numpy as np

model = cv2.dnn.readNet('models/corona_yolov3_1300.weights','models/corona_yolov3.cfg')

claases = ['coronavirus']

layers = model.getLayerNames()

outputLayers = [layers[i[0] - 1] for i in model.getUnconnectedOutLayers()]

test_img = cv2.imread('1.jpg')
test_img = cv2.resize(test_img, None, fx=0.8, fy=0.8)
height, width, _ = test_img.shape

features = cv2.dnn.blobFromImage(test_img, 0.00392, (608, 608), (0,0,0), True, crop=False)

model.setInput(features)
output = model.forward(outputLayers)

boxes = []
confidences = []
class_ids = []

for o in output:
    for detection in o:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            centerX = int(detection[0] * width)
            centerY = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            
            x, y = int(centerX - w / 2), int(centerY - h / 2) 
            
            boxes.append([x,y,w,h]) # pass coordinates
            class_ids.append(class_id) # pass corresponding class
            confidences.append(float(confidence)) # pass corresponding confidence class
    
indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

for i in range(len(boxes)):
    if i in indices:
        x,y,w,h = boxes[i]
        cv2.rectangle(test_img, (x,y), (x + w, y + h), (0,255,255), 2)
        label = claases[class_ids[i]]
        confidence = round(confidences[i] * 100, 2)
        cv2.putText(test_img, label + ' ' + str(confidence) + '%', (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 1)


cv2.imwrite('covid19.png', test_img)
cv2.imshow('covid19 detection.png', test_img)
cv2.waitKey(0)
cv2.destroyAllWindows()