FROM python:3.10

# Prevent interactive tz prompt
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies for OpenCV & RTSP
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

# Expose the Flask port
EXPOSE 5001

# Download YOLO model before start (optional but recommended)
RUN python3 - <<EOF
from ultralytics import YOLO
YOLO('yolov8n.pt')
EOF

CMD ["gunicorn", "--bind", "0.0.0.0:5001", "app:app"]
