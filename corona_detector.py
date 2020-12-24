import cv2
import numpy as np

'''
==================
load yolo model
==================
'''
model = cv2.dnn.readNet('models/corona_yolov3_1300.weights','models/corona_yolov3.cfg')

claases = ['coronavirus','a']

'''
with open('coco.names', 'r') as f:
    claases = [line.strip() for line in f.readlines()]
'''


#  Get the name of all layers of the network
# ============================================
layers = model.getLayerNames()
'''
=======================================
 It gives you the final layers number 
 in the list from net.getLayerNames(). 
 I think it gives the layers number 
 that are unused (final layer). For 
 yolov3, it gave me three number, 
 200, 227, 254. To get the corresponding 
 indexes, we need to do 
 layer_names[i[0] - 1]
 =======================================
'''
outputLayers = [layers[i[0] - 1] for i in model.getUnconnectedOutLayers()]

test_img = cv2.imread('11.jpg')
test_img = cv2.resize(test_img, None, fx=0.8, fy=0.8)
height, width, _ = test_img.shape

'''
=======================
cv2 blobs extracts the
features from an image
img size: (320x320), (416x416), (609x609)
===========================================
'''
features = cv2.dnn.blobFromImage(test_img, 0.00392, (416, 416), (0,0,0), True, crop=False)
'''
for f in features:
    for i,img in enumerate(f):
        cv2.imshow('feature {}'.format(i + 1), img)
'''

# input into neural network
# =========================
model.setInput(features)
output = model.forward(outputLayers)

# showing info on the screen
# ========================== 
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
            
            # rectangle coordinates from center
            # =================================
            x, y = int(centerX - w / 2), int(centerY - h / 2) 
            
            boxes.append([x,y,w,h]) # pass coordinates
            class_ids.append(class_id) # pass corresponding class
            confidences.append(float(confidence)) # pass corresponding confidence class
    
# draw rectangle & put label in each feature in image
# ===================================================
indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

for i in range(len(boxes)):
    if i in indices:
        x,y,w,h = boxes[i]
        cv2.rectangle(test_img, (x,y), (x + w, y + h), (0,255,255), 2)
        label = claases[class_ids[i]]
        confidence = round(confidences[i] * 100, 2)
        cv2.putText(test_img, label + ' ' + str(confidence) + '%', (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 51, 0), 2)
        #print(label, '=>', confidence)   # what kind of features we got from image


cv2.imshow('object detection', test_img)
cv2.waitKey(0)
cv2.destroyAllWindows()