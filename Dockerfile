# Sử dụng image Python 3.9
FROM python:3.9

# Cài đặt các gói hệ thống cần thiết
RUN apt-get update && apt-get install -y cmake

# Đặt thư mục làm việc trong container
WORKDIR /app

# Sao chép tệp requirements.txt vào thư mục làm việc
COPY requirements.txt .

# Cài đặt các thư viện từ requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép mã nguồn của bạn vào thư mục làm việc
COPY . .

# Đặt biến môi trường để Flask chạy trên tất cả các giao diện mạng và trên cổng 8082
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=8082

# Mở cổng 8082 để Flask có thể truy cập từ bên ngoài container
EXPOSE 8082

# Chạy Flask khi container khởi động
CMD ["flask", "run"]
