# importing libraries
import cv2
import numpy as np

# capturing from video source
cap = cv2.VideoCapture("03 - Pistol_.mp4")

# Width and Height of network's input image
whT = 320
# Object detection confidence threshold
confThreshold = 0.5
# Non-maximum Suppression threshold
nmsThreshold = 0.3

# ======== specifying our class name ==========
classNames = ["Pistol"]

# ==== specifying paths for model configuration and weights files  ==============
modelConfiguration = "yolov3-custom.cfg"
modelWeights = "yolov3-custom_2000.weights"

# Creating YOLO Network with the help of configuration and weight files =============
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
# declaring that we are using opencv as backend
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
# declaring to use CPU
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


# making a function to find the objects and draw bounding boxes over the objects
def findObjects(outputs, img):

    hT, wT, cT = img.shape  # Height , Width, and Channels of image
    boundingBoxes = []  # list to store bounding boxes
    classIndexes = []   # list to store classes indexes
    confidenceValues = []   # list to store confidence values

    # ##### We have multiple outputs from the Yolo network Layers
    # ##### so we will loop through single output one by one
    for output in outputs:

        # #### Now we have multiple detections in each single output
        # #### So we will loop through each detection one by one
        for detection in output:

            # ## First 5 out of 6 elements in detection are (cx, cy, w, h, conf)
            # ## And the Other values are the probability of element having object present in the that image
            probScores = detection[5:]  # All the probability score
            classIndex = np.argmax(probScores)  # Index of High probability value
            confidence = probScores[classIndex]  # Getting the High probability Score through the index

            # If the confidence value of having the particular object in the image >= conf Threshold
            # then we will save its Bounding boxes and confidence value and its name
            if confidence >= confThreshold:

                # The original values are in float, so we are converting it to pixels values
                w, h = int(detection[2]*wT), int(detection[3]*hT)
                # These values are the center points not the X, and y coordinates
                # So we are subtracting half of the image-width, image-height value to get the x, y
                x, y = int((detection[0]*wT)-w/2), int((detection[1]*hT)-h/2)

                # appending the values
                boundingBoxes.append([x, y, w, h])
                classIndexes.append(classIndex)
                confidenceValues.append(float(confidence))

    # None Maximum Suppression function remove the overlapping boxes which appears over one object
    # And keep only one box which have maximum confidence score
    # if you are facing the boxes overlapping problem reduce nmsThreshold
    indices = cv2.dnn.NMSBoxes(boundingBoxes, confidenceValues, confThreshold, nmsThreshold)

    # Drawing only those Bounding boxes which is filtered by NMS
    for i in indices:
        box = boundingBoxes[i]
        x, y, w, h = box[0], box[1], box[2], box[3]

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.rectangle(img, (x - 1, y - 25), (x + w+1, y), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, f'{classNames[classIndexes[i]].upper()} {int(confidenceValues[i] * 100)}%',
                    (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


# we will create a while-loop to get the frames from the video
while cap.isOpened():

    # reading frames
    success, frame = cap.read()

    # if the images is read successfully, we will move on
    if success:

        #  #### We cannot input plain image to the network,
        #  #### Yolo net accept only a particular type of image called blob
        # Converting image to blob format
        blob = cv2.dnn.blobFromImage(frame, 1 / 255, (whT, whT), [0, 0, 0], swapRB=True, crop=False)

        # Input blob image to the Yolo network
        net.setInput(blob)

        #  #### In order to get the output from the Yolo network
        #  #### We need to know the names of the output layers
        #  #### So that we can refer to the network for extracting outputs
        # getting the names of output layers of the network
        outputLayersNames = net.getUnconnectedOutLayersNames()
        print(f"Output Layers Name: {outputLayersNames}")

        # forwarding output layers names to the network and storing output of the layers
        # and from this output we will find the bounding boxes
        outputs = net.forward(outputLayersNames)

        # Details of Yolo-Net's OutPut layers
        print(f"Total Number of output layers: {len(outputs)}")
        print(f"Shape of 1st output layer: {outputs[0].shape}")
        print(f"Shape of 2nd output layer: {outputs[1].shape}")
        print(f"Shape of 3rd output layer: {outputs[2].shape}")
        print(f"Inside details of 1st output layer: {outputs[0][0]}")

        # Calling the findObjects Function
        findObjects(outputs, frame)

        # Displaying the footage
        cv2.imshow('Pistol Detection', frame)
        # delaying the frame for 1 milli second
        # and put a condition if press q on screen the detection will be stoped
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        print("Did not read the frame")
        
