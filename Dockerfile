# ------------------------------------------------------------
# ✅ Base Image
# ------------------------------------------------------------
FROM python:3.10-bullseye

# ------------------------------------------------------------
# ✅ System Dependencies (for OpenCV, MediaPipe, aiortc, PyAV)
# ------------------------------------------------------------
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libatlas-base-dev \
    ffmpeg \
    pkg-config \
    libavformat-dev \
    libavcodec-dev \
    libavdevice-dev \
    libavfilter-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    libv4l-dev \
    libssl-dev \
    libffi-dev \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# ✅ Working Directory
# ------------------------------------------------------------
WORKDIR /app

# ------------------------------------------------------------
# ✅ Copy dependencies first (for layer caching)
# ------------------------------------------------------------
COPY requirements.txt .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# ------------------------------------------------------------
# ✅ Copy all source files
# ------------------------------------------------------------
COPY . .

# ------------------------------------------------------------
# ✅ Expose the correct port (Railway = 8080)
# ------------------------------------------------------------
EXPOSE 8080

# ------------------------------------------------------------
# ✅ Start Flask app with Gunicorn (production)
# ------------------------------------------------------------
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]
