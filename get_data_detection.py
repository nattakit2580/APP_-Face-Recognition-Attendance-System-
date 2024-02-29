import cv2
from ultralytics import YOLO
import datetime as dt


model = YOLO(r'C:\Users\asus\Desktop\APP\APP\yolov8n-face (1).pt')
cap = cv2.VideoCapture(1)
all_classes = model.names
print(all_classes)

num = 0

while True:
    ret, frame = cap.read()
    results = model.predict(source=frame,
                            device='cpu',
                            conf=0.51,
                            iou=0.1,
                            # classes = [],
                            imgsz = (640,640))

    print(results)

    for result in results:
        boxes = result.boxes.xyxy.tolist()
        classes = result.boxes.cls.tolist()
        names = result.names
        confidences = result.boxes.conf.tolist()
        for box, cls, conf in zip(boxes, classes, confidences):
            x1, y1, x2, y2 = box
            x1 = int(x1 / frame.shape[1] * frame.shape[1])
            x2 = int(x2 / frame.shape[1] * frame.shape[1])
            y1 = int(y1 / frame.shape[0] * frame.shape[0] )
            y2 = int(y2 / frame.shape[0] * frame.shape[0] )

            roi = frame[y1:y2, x1:x2]

            # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) 
            cv2.putText(frame, f"{names[int(cls)]}: {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    cv2.imshow('Object Detection', frame)
    cv2.imshow('roi', roi)

    if cv2.waitKey(1) == ord('q'):
        cv2.imwrite(r'C:\Users\asus\Desktop\APP\APP\image_test\sai' + str(num) + '.jpg', roi)
        num += 1
        # break

    if cv2.waitKey(1) & 0xFF == ord('1'):
        break
        

cap.release()
cv2.destroyAllWindows()