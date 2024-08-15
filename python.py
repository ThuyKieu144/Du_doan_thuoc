import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Đọc dữ liệu từ tệp CSV
data = pd.read_csv('healthcare_dataset.csv')

# Kiểm tra và xử lý các giá trị bị thiếu
data = data.dropna(subset=['Medication'])

# Mã hóa các giá trị phân loại thành số
le_medical_condition = LabelEncoder()
le_test_results = LabelEncoder()
le_admission_type = LabelEncoder()
le_medication = LabelEncoder()
le_gender = LabelEncoder()

data['Medical Condition'] = le_medical_condition.fit_transform(data['Medical Condition'])
data['Test Results'] = le_test_results.fit_transform(data['Test Results'])
data['Admission Type'] = le_admission_type.fit_transform(data['Admission Type'])
data['Medication'] = le_medication.fit_transform(data['Medication'])
data['Gender'] = le_gender.fit_transform(data['Gender'])

# Chọn các đặc trưng và mục tiêu
X = data[['Age', 'Gender', 'Medical Condition', 'Test Results', 'Admission Type']]
y = data['Medication']

# Xử lý các giá trị bị thiếu trong các đặc trưng
imputer = SimpleImputer(strategy='most_frequent')
X = imputer.fit_transform(X)

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Huấn luyện mô hình hồi quy logistic
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred = model.predict(X_test)

# Lưu mô hình đã huấn luyện và các bộ mã hóa
with open('medication_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('label_encoders.pkl', 'wb') as le_file:
    pickle.dump({
        'medical_condition': le_medical_condition,
        'test_results': le_test_results,
        'admission_type': le_admission_type,
        'medication': le_medication,
        'gender': le_gender,
    }, le_file)
