# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install system dependencies untuk PyTorch
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy aplikasi
COPY . .

# Buat direktori untuk template
RUN mkdir -p templates

# Expose port
EXPOSE 8000

# Run aplikasi
CMD ["fastapi", "app:app", "--host", "0.0.0.0", "--port", "8000"]