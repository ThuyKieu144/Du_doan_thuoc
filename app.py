from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Tải mô hình đã huấn luyện và các bộ mã hóa
with open('medication_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('label_encoders.pkl', 'rb') as le_file:
    le = pickle.load(le_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Chuyển đổi dữ liệu từ biểu mẫu thành danh sách đặc trưng
        age = float(request.form['age'])
        gender = request.form['gender']
        medical_condition = request.form['medical_condition']
        test_results = request.form['test_results']
        admission_type = request.form['admission_type']

        # Chuyển đổi các giá trị văn bản thành số bằng cách sử dụng mã hóa
        gender_encoded = int(gender)  # Đảm bảo giới tính được chuyển đổi thành số nguyên
        medical_condition_encoded = le['medical_condition'].transform([medical_condition])[0]
        test_results_encoded = le['test_results'].transform([test_results])[0]
        admission_type_encoded = le['admission_type'].transform([admission_type])[0]

        # Tạo mảng numpy từ các đặc trưng
        final_features = np.array([age, gender_encoded, medical_condition_encoded, test_results_encoded, admission_type_encoded]).reshape(1, -1)

        # Dự đoán thuốc
        prediction = model.predict(final_features)

        # Chuyển đổi dự đoán trở lại giá trị văn bản gốc
        medication_prediction = le['medication'].inverse_transform([prediction[0]])[0]

        # Trả kết quả dự đoán về giao diện người dùng
        return render_template('index.html', prediction_text='Dự đoán thuốc uống: {}'.format(medication_prediction))

    except Exception as e:
        # Xử lý lỗi trong việc chuyển đổi dữ liệu đầu vào
        print(f'Lỗi: {e}')  # In lỗi ra console để gỡ lỗi
        return render_template('index.html', prediction_text='Dữ liệu đầu vào không hợp lệ. Vui lòng kiểm tra lại.')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
