from flask import Flask, render_template, request, jsonify
from PIL import Image
import torch
from torchvision.models import resnet101
from torchvision import transforms
import pyodbc
import numpy as np
from ultralytics import YOLO
import datetime
import os
import cv2

app = Flask(__name__)

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
            output_file_path = r'C:\Users\asus\Desktop\APP\APP\output_recognition_data.pth'
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

# ฟังก์ชันสำหรับการระบุตัวบุคคลในภาพ
def recognize_face(img_input):
    # โหลดโมเดล ResNet101
    model = resnet101(pretrained=True)
    model.eval()

    # โหลดข้อมูลจากไฟล์ .pth
    recognition_data_path = r'C:\Users\asus\Desktop\APP\APP\recognition_data.pth'
    recognition_data = torch.load(recognition_data_path, map_location=torch.device('cpu'))

    transform = transforms.Compose([
            transforms.Resize((200, 100)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.526,0.510,0.560], std=[0.128,0.127,0.126]),
    ])

    img_input_tensor = transform(img_input)
    img_input_tensor = torch.unsqueeze(img_input_tensor, 0)

    with torch.no_grad():
        feat_input = model(img_input_tensor)

    max_similarity = -1.0
    best_match_name = ""
    best_match_data = None

    # โหลดข้อมูลจากฐานข้อมูล
    server = 'LAPTOP-3S4KNO5G\SQLEXPRESS01'
    database = 'Pythonapp'
    driver = '{SQL Server}'
    trusted_connection = 'yes'
    connection_string = f'DRIVER={driver};SERVER={server};DATABASE={database};Trusted_Connection={trusted_connection}'

    connection = pyodbc.connect(connection_string)
    cursor = connection.cursor()

    cursor.execute('SELECT TOP (1000) [ID], [Name], [Position], [create_date], [emp_ID] FROM [Pythonapp].[dbo].[Application]')
    employees = cursor.fetchall()

    for emp in employees:
        emp_id, emp_name, emp_position, emp_create_date, emp_emp_id = emp
        feat_db = recognition_data[emp_id]['feature']
        similarity = torch.nn.functional.cosine_similarity(feat_input, feat_db)
        similarity = similarity.item() * 100

        if similarity > max_similarity:
            max_similarity = similarity
            best_match_name = emp_name
            best_match_data = {"emp_ID": emp_emp_id, "Name": emp_name, "Position": emp_position, "create_date": emp_create_date}

    if max_similarity <= 85:
        best_match_name = "Unknown"

    cursor.close()
    connection.close()

    return best_match_name, max_similarity, best_match_data

def update_database(emp_id, time_column, new_time):
    try:
        server = 'LAPTOP-3S4KNO5G\SQLEXPRESS01'
        database = 'Pythonapp'
        driver = '{SQL Server}'
        trusted_connection = 'yes'
        connection_string = f'DRIVER={driver};SERVER={server};DATABASE={database};Trusted_Connection={trusted_connection}'

        connection = pyodbc.connect(connection_string)
        cursor = connection.cursor()

        query = f"UPDATE [Pythonapp].[dbo].[Web_inout] SET {time_column} = ? WHERE emp_ID = ?"

        cursor.execute(query, (new_time, emp_id))
        connection.commit()

        print(f"Updated {time_column} successfully.")

    except pyodbc.Error as db_error:
        print(f"Database error: {db_error}")
    except Exception as e:
        print(f"General error: {e}")
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'connection' in locals():
            connection.close()

@app.route('/')
def index():
    return render_template('index.html')

crop_dir = r'C:\Users\asus\Desktop\APP\APP\static\crop'

def check_and_rotate(image):
    width, height = image.size
    if width > height:
        return image.rotate(270, expand=True)
    else:
        return image

# ในฟังก์ชัน upload()
@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return render_template('index.html', result="No file part")

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', result="No selected file")

    if file:
        # อ่านภาพและเปลี่ยนเป็นภาพ numpy array
        img = Image.open(file)
        img_np = np.array(img)

        # โหลดโมเดล YOLOv5 เพื่อตรวจจับหน้าในภาพ
        model = YOLO(r'C:\Users\asus\Desktop\APP\APP\yolov8n-face.pt')
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

                    face_img = Image.fromarray(roi)

                    # ตรวจสอบและหมุนภาพให้เป็นแนวตั้ง (vertical)
                    face_img = check_and_rotate(face_img)

                    name, similarity, data = recognize_face(face_img)

                    # สร้างวัตถุ datetime สำหรับเก็บเวลาปัจจุบัน
                    current_time = datetime.datetime.now()

                    # แปลงเวลาเป็นรูปแบบที่ต้องการ เช่น 'YYYY-MM-DD HH:MM:SS'
                    formatted_time = current_time.strftime('%Y-%m-%d_%H-%M-%S')

                    # สร้าง directory ถ้ายังไม่มี
                    if not os.path.exists(crop_dir):
                        os.makedirs(crop_dir)

                    # บันทึกรูปที่ crop ไว้
                    crop_path = os.path.join(crop_dir, f'{name}_{formatted_time}.jpg')
                    face_img.save(crop_path)

                    # ตรวจสอบเงื่อนไขและอัปเดตค่าในคอลัมน์ที่เหมาะสม
                    if current_time.hour >= 6 and current_time.hour < 12:
                        print("เวลาอยู่ในช่วงขาเข้า")
                        formatted_time = datetime.datetime.strptime(formatted_time, '%Y-%m-%d_%H-%M-%S')
                        update_database(data["emp_ID"], time_column='Time_in', new_time=formatted_time)

                        return render_template('result.html', name=name, similarity=similarity, data=data, time=formatted_time, crop_path=crop_path.split('APP\\')[-1])

                    if current_time.hour >= 12 and current_time.hour < 22:
                        print("เวลาอยู่ในช่วงขาออก")
                        formatted_time = datetime.datetime.strptime(formatted_time, '%Y-%m-%d_%H-%M-%S')
                        update_database(data["emp_ID"], time_column='Time_out', new_time=formatted_time)

                        return render_template('result.html', name=name, similarity=similarity, data=data, time=formatted_time, crop_path=crop_path.split('APP\\')[-1])

        else:
            return render_template('index.html', result="No faces detected")

    return render_template('index.html', result="File uploaded successfully")

# ปุ่ม In
@app.route('/scan_in', methods=['POST'])
def scan_in():
    # ตรวจสอบและสแกนเข้างาน
    # คืนข้อความแสดงผลการสแกนเข้างาน
    result = "Scanned in successfully."
    return render_template('index.html', result=result)

# ปุ่ม Out
@app.route('/scan_out', methods=['POST'])
def scan_out():
    # ตรวจสอบและสแกนออกงาน
    # คืนข้อความแสดงผลการสแกนออกงาน
    result = "Scanned out successfully."
    return render_template('index.html', result=result)

if __name__ == "__main__":
    fetch_data_from_db()  # ดึงข้อมูลจากฐานข้อมูล
    app.run(debug=True)
