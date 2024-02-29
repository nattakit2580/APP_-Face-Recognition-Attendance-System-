from flask import Flask, render_template, request, jsonify
from PIL import Image
import torch
from torchvision.models import resnet50
from torchvision import transforms
import pyodbc
from io import BytesIO
import base64

app = Flask(__name__)

# ฟังก์ชันสำหรับการเชื่อมต่อฐานข้อมูลและดึงข้อมูล
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
        cursor.execute('SELECT TOP (1000) [pth] FROM [Pythonapp].[dbo].[pythorch]')
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

# ฟังก์ชันสำหรับการระบุตัวบุคคลในภาพ
def recognize_face(img_input):
    # โหลดโมเดล ResNet50
    model = resnet50(pretrained=True)
    model.eval()

    # โหลดข้อมูลจากไฟล์ .pth
    recognition_data_path = r"C:\Users\asus\Desktop\APP\APP\recognition_data.pth"
    recognition_data = torch.load(recognition_data_path, map_location=torch.device('cpu'))

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

    if max_similarity <= 75:
        best_match_name = "Unknown"

    cursor.close()
    connection.close()

    return best_match_name, max_similarity, best_match_data


# หน้าแรกของเว็บไซต์
@app.route('/')
def index():
    return render_template('index.html')

# การอัปโหลดภาพและประมวลผล
@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return render_template('index.html', result="No file part")

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', result="No selected file")

    if file:
        img = Image.open(file)
        name, similarity, data = recognize_face(img)
        if data:
            return render_template('result.html', name=name, similarity=similarity, data=data)
        else:
            return render_template('result.html', name=name, similarity=similarity, data=None)

# หน้าที่ใช้ในการถ่ายภาพด้วยกล้อง
@app.route('/capture', methods=['POST'])
def capture():
    # ตรวจสอบว่าคีย์ 'image_data' มีอยู่ใน request.form หรือไม่
    if 'image_data' not in request.form:
        return jsonify(error="Missing 'image_data' key in request form")

    # รับภาพจากฟอร์มและแปลงเป็นรูปภาพ PIL
    image_data = request.form['image_data']
    image = Image.open(BytesIO(base64.b64decode(image_data.split(',')[1])))

    # ทำนายภาพ
    name, similarity, data = recognize_face(image)

    # แสดงผลลัพธ์
    if data:
        return render_template('result.html', name=name, similarity=similarity, data=data)
    else:
        return render_template('result.html', name=name, similarity=similarity, data=None)


if __name__ == "__main__":
    fetch_data_from_db()  # ดึงข้อมูลจากฐานข้อมูล
    app.run(debug=True)
    
#fgfg