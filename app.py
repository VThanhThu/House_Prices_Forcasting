import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Định nghĩa mô hình phân loại
class HousePriceModel(nn.Module):
    def __init__(self):
        super(HousePriceModel, self).__init__()
        self.layer1 = nn.Linear(13, 64)  # 13 đặc trưng đầu vào
        self.bn1 = nn.BatchNorm1d(64)
        self.layer2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.layer3 = nn.Linear(32, 16)
        self.bn3 = nn.BatchNorm1d(16)
        self.out = nn.Linear(16, 3)  # Đầu ra có 3 lớp: Cao, Trung bình, Thấp

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.relu(self.bn1(self.layer1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.layer2(x)))
        x = self.dropout(x)
        x = self.relu(self.bn3(self.layer3(x)))
        x = self.out(x)
        return F.softmax(x, dim=1)  # Sử dụng softmax để ra xác suất của 3 lớp

# Load mô hình đã train
model = HousePriceModel()
model.load_state_dict(torch.load("D:\\House Prices _ onehot\\src\\model\\house_price_model.pth", map_location=torch.device('cpu')))
model.eval()

# Giao diện Streamlit
st.title("🏠 HOUSE PRICE FORECASTING")

# Nhập dữ liệu đầu vào
carpet_area = st.number_input("Area (sqft)", min_value=10.0, max_value=100000000.0, step=1.0)
current_floor = st.number_input("Current Floor", min_value=1, max_value=50, step=1)
total_floor = st.number_input("Total Floor", min_value=1, max_value=80, step=1)
bathroom = st.number_input("Bathroom", min_value=0, max_value=20, step=1)

# Các trường bổ sung:
balcony = st.number_input("Balcony", min_value=0, max_value=20, step=1)
car_parking = st.number_input("Car Parking", min_value=0, max_value=20, step=1)

# Các trường loại hình nhà (Types) - Tách thành 3 input
types_covered = st.selectbox("Is the property covered?", options=["Yes", "No"])
types_open = st.selectbox("Is the property open?", options=["Yes", "No"])
types_unknown = st.selectbox("Is the property type unknown?", options=["Yes", "No"])

# Các trường về tình trạng nội thất (Furnishing) - Tách thành 3 input
furnishing_furnished = st.selectbox("Is the property furnished?", options=["Yes", "No"])
furnishing_semi_furnished = st.selectbox("Is the property semi-furnished?", options=["Yes", "No"])
furnishing_unfurnished = st.selectbox("Is the property unfurnished?", options=["Yes", "No"])
furnishing_unkonw = st.selectbox("Is the property unknow?", options=["Yes", "No"])

# Chuyển đổi dữ liệu từ lựa chọn về dạng số để dễ xử lý
types_value = [
    1 if types_covered == "Yes" else 0,
    1 if types_open == "Yes" else 0,
    1 if types_unknown == "Yes" else 0
]

furnishing_value = [
    1 if furnishing_furnished == "Yes" else 0,
    1 if furnishing_semi_furnished == "Yes" else 0,
    1 if furnishing_unfurnished == "Yes" else 0,
    1 if furnishing_unkonw == "Yes" else 0
]

# Khi nhấn nút "Forecasting"
if st.button("Forecasting"):
    # Dữ liệu đầu vào gồm các đặc trưng số và các đặc trưng đã chuyển đổi sang giá trị số
    input_data = torch.tensor(
        [[carpet_area, current_floor, total_floor, bathroom, balcony, car_parking] + types_value + furnishing_value], 
        dtype=torch.float32
    )
    
    # Dự đoán giá nhà (model phải được load trước đó)
    predicted_probabilities = model(input_data).detach().numpy().flatten()  # Lấy xác suất từ mô hình
    
    # Dựa trên xác suất, chọn lớp có xác suất cao nhất
    predicted_class = np.argmax(predicted_probabilities)

    # Tạo tên cho các lớp dựa trên chỉ số lớp
    class_labels = ["Low", "Medium", "High"]
    predicted_label = class_labels[predicted_class]

    # Hiển thị kết quả phân loại mà không cần hiển thị xác suất
    st.success(f"💰 Estimated house price is: {predicted_label}")
