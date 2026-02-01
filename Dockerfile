FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Cấp quyền thực thi cho script khởi động
RUN chmod +x start.sh

# Create data directories
RUN mkdir -p /app/data/uploads /app/data/processed_data /app/data/raw_data

# Expose port
EXPOSE 10000

# Thay đổi lệnh CMD để chạy script thay vì uvicorn trực tiếp
CMD ["./start.sh"]