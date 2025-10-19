FROM python:3.10-bullseye

# --- Install system dependencies (for OpenCV, MediaPipe, TensorFlow, aiortc, PyAV)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libatlas-base-dev \
    ffmpeg \
    pkg-config \
    libavformat-dev \
    libavcodec-dev \
    libavdevice-dev \# ------------------------------------------------------------
# ✅ Base Image
# ------------------------------------------------------------
FROM python:3.10-bullseye

# ------------------------------------------------------------
# ✅ System Dependencies (for OpenCV + aiortc + PyAV)
# ------------------------------------------------------------
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    libavdevice-dev \
    libavfilter-dev \
    libswscale-dev \
    libavformat-dev \
    libavcodec-dev \
    libavutil-dev \
    libv4l-dev \
    libssl-dev \
    libffi-dev \
    build-essential \
    pkg-config \
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

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# ------------------------------------------------------------
# ✅ Copy all source files
# ------------------------------------------------------------
COPY . .

# ------------------------------------------------------------
# ✅ Expose the correct port (Railway uses 8080)
# ------------------------------------------------------------
EXPOSE 8080

# ------------------------------------------------------------
# ✅ Start the app using Gunicorn (recommended for production)
# ------------------------------------------------------------
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]

    libavutil-dev \
    libavfilter-dev \
    libswscale-dev \
    libswresample-dev \
    libv4l-dev \
    libssl-dev \
    libffi-dev \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# --- Work directory
WORKDIR /app

# --- Copy dependencies first for caching
COPY requirements.txt .

# --- Upgrade pip and install requirements
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir --verbose -r requirements.txt

# --- Copy all source files
COPY . .

# --- Expose default HF Space port
EXPOSE 7860

# --- Start Flask app
CMD ["python", "app.py"]
