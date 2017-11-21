
# coding: utf-8

# # Demo of loading a Caffe Deep Model using OpenCV on Predix
# 
# by Dattaraj J Rao - Principal Architect - GE Transportation - <a href='mailto:Dattaraj.Rao@ge.com'>Dattaraj.Rao@ge.com</a>
# 
# code inspired by Adrian Rosebrock's post at: https://www.pyimagesearch.com/2017/08/21/deep-learning-with-opencv/

# ## Imports needed

# In[2]:


import numpy as np
import cv2

# Model parameters
MODEL_PROTO = "Mobilenet-SSD/deploy.prototxt"
MODEL_FILE = "Mobilenet-SSD/MobileNetSSD_deploy.caffemodel"
MODEL_LABELS = "Mobilenet-SSD/labels.txt"
MODEL_INPUT_SIZE = (300,300)

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(MODEL_PROTO, MODEL_FILE)

# load the class labels from disk
rows = open(MODEL_LABELS).read().strip().split("\n")
CLASSES = [r[r.find(" ") + 1:].split(",")[0] for r in rows]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

print("[INFO] Model loaded successfully")

# In[20]:

def run_model(image_np):
    # define the reurn array
    retimages = {}
    # check image
    if image_np is None:
        return retimages
    # resize image
    image_np = cv2.resize(image_np, MODEL_INPUT_SIZE)
    (h, w) = image_np.shape[:2]
    # create a conv net friendly image with mean averaging and scaling 
    blob = cv2.dnn.blobFromImage(image_np, scalefactor = (1/127.5), size = MODEL_INPUT_SIZE, mean = 127.5)
    # pass this blob as input to the network
    net.setInput(blob)
    detections = net.forward()
	
    # loop over the detections
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the prediction
        confidence = detections[0, 0, i, 2]
        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > 0.5:
            # extract the index of the class label from the `detections`,
            # then compute the (x, y)-coordinates of the bounding box for
            # the object
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # display the prediction
            label = "{}_{:.2f}%".format(CLASSES[idx], confidence * 100)            
            cv2.rectangle(image_np, (startX, startY), (endX, endY), (255,255,0), 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(image_np, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,0), 2)

    return image_np

# ## Make the Prediction using forward pass

# In[21]:
