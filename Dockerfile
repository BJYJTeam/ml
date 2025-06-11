FROM python:3.10-slim

# 필수 시스템 패키지 설치
RUN apt update && apt install -y \
    gcc \
    default-libmysqlclient-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN apt-get update && apt-get install -y \
    default-libmysqlclient-dev \
    pkg-config \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
