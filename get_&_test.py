import pyodbc
import torch
from torchvision.models import resnet50
from torchvision import transforms
from PIL import Image
import os
import cv2
import numpy as np
from ultralytics import YOLO

# รวมส่วนของการเชื่อมต่อฐานข้อมูลและการดึงข้อมูล
def fetch_data_from_db():
    try:
        # กำหนดค่าการเชื่อมต่อ
        server = 'LAPTOP-3S4KNO5G\SQLEXPRESS01'
        database = 'Pythonapp'
        driver = '{SQL Server}'
        trusted_connection = 'yes'
        connection_string = f'DRIVER={driver};SERVER={server};DATABASE={database};Trusted_Connection={trusted_connection}'

        # เชื่อมต่อฐานข้อมูล
        connection = pyodbc.connect(connection_string)
        cursor = connection.cursor()

        # ดึงข้อมูลจากคอลัมน์ 'pth'
        cursor.execute('SELECT pth FROM dbo.pythorch')
        row = cursor.fetchone()

        if row:
            binary_data = row[0]
            output_file_path = r'C:\Users\asus\Desktop\APP\APP\recognition_data.pth'
            with open(output_file_path, 'wb') as file:
                file.write(binary_data)
            print(f"เขียนข้อมูลลงไฟล์ {output_file_path} สำเร็จ")
        else:
            print("ไม่พบข้อมูล")

    except pyodbc.Error as db_error:
        print(f"ข้อผิดพลาดฐานข้อมูล: {db_error}")
    except Exception as e:
        print(f"ข้อผิดพลาดทั่วไป: {e}")
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'connection' in locals():
            connection.close()

# รวมส่วนของการโหลดโมเดลและประมวลผลภาพ
def recognize_face(img_input):
    # โหลดโมเดล ResNet50
    model = resnet50(pretrained=True)
    model.eval()

    # โหลดข้อมูลจากไฟล์ .pth
    recognition_data_path = r"C:\Users\asus\Desktop\APP\APP\recognition_data.pth"
    recognition_data = torch.load(recognition_data_path)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img_input_tensor = transform(img_input)
    img_input_tensor = torch.unsqueeze(img_input_tensor, 0)

    with torch.no_grad():
        feat_input = model(img_input_tensor)

    max_similarity = -1.0
    best_match_name = ""

    for data in recognition_data:
        feat_db = data['feature']
        person_name = data['image']

        similarity = torch.nn.functional.cosine_similarity(feat_input, feat_db)
        similarity = similarity.item()*100

        
        if similarity > max_similarity:
             max_similarity = similarity
             best_match_name = person_name
             best_match_name = best_match_name.split('_')[1]
             

    if max_similarity <= 75:
        best_match_name = "Unknown"

    text = f"employee ID: {os.path.splitext(best_match_name)[0]}, Score: {max_similarity:.4f}"
    print(text)

    # เพิ่มข้อความลงในภาพ (การแสดงผลในส่วนนี้อาจต้องปรับตามความต้องการ)
    img_input_with_text = np.array(img_input)
    img_input_with_text = cv2.putText(img_input_with_text, text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
    img_input_with_text = cv2.resize(cv2.cvtColor(img_input_with_text, cv2.COLOR_BGR2RGB), (369, 493))

    # หมายเหตุ: คุณอาจต้องปรับเปลี่ยนการแสดงผลให้เหมาะสมกับสภาพแวดล้อมที่คุณใช้งาน
    cv2.imshow("Input Image", img_input_with_text)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return img_input_with_text



model = YOLO(r'C:\Users\asus\Desktop\APP2\yolov8n-face.pt')
cap = cv2.VideoCapture(0)
while True:
    ret,frame = cap.read()

    # img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = frame
    
    results = model.predict(source=img,
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
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

    pil_image = Image.fromarray(roi)
    # img_input = Image.open(img_input_path)
    print(img)
    cv2.imshow("Input image", img.copy())
    
    key = cv2.waitKey(1)
    if key == ord('q'):
        recognize_face(pil_image)
        break
    
cv2.destroyAllWindows()

if __name__ == "__main__":
    fetch_data_from_db()  # ดึงข้อมูลจากฐานข้อมูล

    img_input_path = r"C:\Users\asus\Desktop\APP\APP\dataset_test\nao\5.jpg"
    img_input = Image.open(img_input_path)
    recognize_face(img_input)
