import torch
from torchvision.models import resnet50
from torchvision import transforms
from PIL import Image
import os
import cv2
import numpy as np

# โหลดโมเดล ResNet50
model = resnet50(pretrained=True)
model.eval()

# ตำแหน่งของไฟล์ .pth ที่บันทึกข้อมูลลักษณะเด่น
recognition_data_path = r"C:\Users\asus\Desktop\APP\APP\recognition_data.pth"

# โหลดข้อมูลจากไฟล์ .pth
recognition_data = torch.load(recognition_data_path)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def recognize_face(img_input):

    img_input_tensor = transform(img_input)
    img_input_tensor = torch.unsqueeze(img_input_tensor, 0)

    with torch.no_grad():
        feat_input = model(img_input_tensor)

    max_similarity = -1.0
    best_match_name = ""

    for data in recognition_data:
        feat_db = data['feature']
        person_name = data['image']

        # เปรียบเทียบคุณลักษณะ
        similarity = torch.nn.functional.cosine_similarity(feat_input, feat_db)

        if similarity > max_similarity:
            max_similarity = similarity
            best_match_name = person_name

    img_input_with_text = np.array(img_input)
    text = f"Name: {os.path.splitext(best_match_name)[0]}, Score: {max_similarity.item():.4f}"
    print(text)

    img_input_with_text = cv2.putText(img_input_with_text, text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
    img_input_with_text = cv2.resize(cv2.cvtColor(img_input_with_text, cv2.COLOR_BGR2RGB),(369,493))

    cv2.imshow("Input Image", img_input_with_text)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 
    return img_input_with_text

img_input_path = r"C:\Users\asus\Desktop\APP\APP\image_test\rock\0.jpg"
img_input = Image.open(img_input_path)
print(type(img_input))
recognize_face(img_input)