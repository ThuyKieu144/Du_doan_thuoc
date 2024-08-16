# Sử dụng image Python chính thức từ Docker Hub
FROM python:3.8-slim

# Đặt thư mục làm việc
WORKDIR /app

# Sao chép nội dung thư mục hiện tại vào container
COPY . /app

# Cài đặt các gói cần thiết
RUN pip install --no-cache-dir -r requirements.txt

# Chạy file python.py để huấn luyện mô hình (chạy trong build process)
RUN python python.py

# Mở port 5000 để truy cập từ ngoài container
EXPOSE 5000

# Khởi động ứng dụng Flask khi container chạy
CMD ["python", "app.py"]
