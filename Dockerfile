FROM python:3.11-slim

# ffmpeg is needed for OpenCV's VideoCapture to decode MP4 and RTSP streams
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY analyze.py .

# Video input is expected at /data/input.mp4 (mount via -v)
VOLUME ["/data"]

ENTRYPOINT ["python", "analyze.py"]
CMD ["--help"]
