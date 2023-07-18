import cv2
import numpy as np
import pyttsx3
import time

# Load YOLOv4 model and configuration
net = cv2.dnn.readNetFromDarknet('yolov4.cfg', 'yolov4.weights')

# Set the classes that the model can detect
classes = []
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Define the colors for drawing bounding boxes
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Get the names of all layers in the network
layer_names = net.getLayerNames()
output_layers = net.getUnconnectedOutLayersNames()

# Initialize text-to-speech engine
engine = pyttsx3.init()

vid = cv2.VideoCapture(0)

# Dictionary to keep track of last announcement time for each object
last_announced = {}

while True:
    ret, frame = vid.read()

    # Perform object detection
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (608, 608), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Initialize bounding box coordinates and class labels
    boxes = []
    confidences = []
    class_ids = []

    # Iterate over each detection
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                # Get the coordinates of the bounding box
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Calculate the top-left corner coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-maximum suppression to eliminate overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding boxes and display class names
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = colors[class_ids[i]]

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Check if object is new or hasn't been announced in the last 5 seconds
            if label not in last_announced or time.time() - last_announced[label] >= 5:
                # Display class label and confidence
                text = f'{label}: {confidence:.2f}'
                cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Speak the object label
                engine.say(label)
                engine.runAndWait()

                # Update the last announcement time for the object
                last_announced[label] = time.time()

    # Display the frame
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
