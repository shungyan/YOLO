import cv2 
import numpy as np

# Load Yolo
model = cv2.dnn.readNet("yolov3/yolov3.weights", "yolov3/yolov3.cfg")
classes = []
with open("yolov3/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

colors=np.random.uniform(0,255,size=(len(classes),3))

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, img=cap.read()

    height, width, channels = img.shape

    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    layer_names = model.getLayerNames()
    output_layers = [layer_names[i - 1] for i in model.getUnconnectedOutLayers()]
    model.setInput(blob)
    outs = model.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
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


    indexes=cv2.dnn.NMSBoxes(boxes,confidences,0.5,0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y + 30), font, 3, color, 3)

    cv2.imshow("Image", img)
    if cv2.waitKey(10) & 0xFF==ord('q'):
        break

cv2.destroyAllWindows()




