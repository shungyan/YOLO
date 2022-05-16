import cv2 
import numpy as np

# Load Yolo
model = cv2.dnn.readNet(model="", config="")
classes = []
with open("", "r") as f:
    classes = [line.strip() for line in f.readlines()]

colors=np.random.uniform(0,255,size=(len(classes),3))

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, img=cap.read()

    height, width, channels = img.shape

    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    model.setInput(blob)

    outputs=model.forward()

    for output in outputs:
        for detection in output:
            confidence=detection[2]

            if confidence>0.5:
                class_id=detection[1]
                class_name=classes[int(class_id)-1]
                color=colors(int(class_id))

                x=detection[3]*width
                y=detection[4]*height
                w=detection[5]*width
                h=detection[6]*height

                cv2.rectangle(img,(int(x),int(y)),(int(w),int(h)),color,thickness=2)   
                cv2.putText(img, class_name, (x, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 3, color, 3)

    cv2.imshow("Image", img)
    if cv2.waitKey(10) & 0xFF==ord('q'):
        break

cv2.destroyAllWindows()




