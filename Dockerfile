# ==============================================================
#  PLANT DISEASE DETECTION — Dockerfile
# ==============================================================
#  Builds a containerized version of the Flask web app.
#  Uses a slim Python base to keep image size small.
#
#  Build : docker build -t plant-disease-detector .
#  Run   : docker run -p 5000:5000 plant-disease-detector
#  Open  : http://localhost:5000
# ==============================================================

# Base image: slim Python 3.10
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Install system dependencies needed by OpenCV / Pillow
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (Docker layer caching — only reinstalls if requirements change)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files into the container
COPY app.py .
COPY model.py .
COPY data_prep.py .
COPY best_model.pth .
COPY class_names.json .
COPY templates/ ./templates/

# Expose port 5000
EXPOSE 5000

# Set environment variables
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

# Run the Flask app
CMD ["python", "app.py"]