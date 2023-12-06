#import thư viện cần thiết
import cv2
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

def main():
    st.set_page_config(page_title="Object Detection App")
    st.title("Object Detection App")

    #tải ảnh
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Read image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        #lấy chiều dài, chiều rộng của ảnh
        (h, w) = image.shape[:2]

        #load object detection model 
        net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt.txt', 'MobileNetSSD_deploy.caffemodel')

        #danh sách đối tượng được phân loại
        categories = { 0: 'background', 1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat', 5: 'bottle', 
                      6: 'bus', 7: 'car', 8: 'cat', 
                  9: 'chair', 10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse', 14: 'motorbike', 15: 'person', 
                  16: 'pottedplant', 17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor'}
        classes = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", 
                   "car", "cat", "chair", "cow", 
                "diningtable",  "dog", "horse", "motorbike", "person", "pottedplant", "sheep", 
                "sofa", "train", "tvmonitor"]

        #nhận diện đối tượng trong ảnh
        #tạo blob cho ảnh 
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
        #đặt đầu vào của mô hình đã được huấn luyện cho ảnh được tiền xử lý
        net.setInput(blob)
        #chạy mô hình đã được huấn luyện và trả về kết quả đầu ra, 
        #bao gồm các hộp giới hạn đối tượng được phát hiện và 
        #điểm độ tin cậy tương ứng
        detections = net.forward()
        #vẽ khung giới hạn và hiển thị kết quả
        #np.random.uniform: tạo ra các màu ngẫu nhiên sẽ được 
        #sử dụng để vẽ khung giới hạn xung quanh các đối tượng được phát hiện
        colors = np.random.uniform(255, 0, size=(len(categories), 3))

        #vẽ bounding boxes và nhãn trên hình
        #lặp qua tất cả các đối tượng được phát hiện và 
        #vẽ một khung giới hạn xung quanh mỗi đối tượng với nhãn tương ứng
        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.2:
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                label = "{}: {:.2f}%".format(classes[idx], confidence * 100) 
                #vẽ khung giới hạn
                cv2.rectangle(image, (startX, startY), (endX, endY), colors[idx], 2)     
                y = startY - 15 if startY - 15>15 else startY + 15     
                #thêm nhãn cho khung giới hạn
                cv2.putText(image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[idx], 2)
        
        #hiển thị hình 
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_column_width=True)
        
if __name__ == '__main__':
    main()
