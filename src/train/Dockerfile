# Use official Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy files into the container
COPY src/train/train.py /app/train.py
COPY data/processed /app/data/processed
COPY requirements.txt /app/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Command to run the training script
CMD ["python", "train.py"]
