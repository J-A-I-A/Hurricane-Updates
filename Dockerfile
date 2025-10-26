# Use the official RunPod PyTorch image as the base
# This image already includes Python, PyTorch, and the necessary CUDA drivers
FROM runpod/pytorch:2.3.0-py3.10-cuda12.1.1-devel-ubuntu22.04

# Set a working directory inside the container
WORKDIR /app

# Copy the requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Install the Python dependencies
# We add --no-cache-dir to keep the image size down
RUN pip install --no-cache-dir -r requirements.txt

# Copy the main application script and any other files
COPY . .

# Set the default command to start the RunPod serverless worker
# This is the same command from your RunPod template
CMD ["python3", "bulletin_ocr_service.py"]