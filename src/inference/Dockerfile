# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy inference script and processed data into the container
COPY inference.py .
COPY data/processed /app/data/processed
COPY outputs/models /app/outputs/models

# Ensure the output directory exists
RUN mkdir -p /app/outputs/predictions

# Run the inference script automatically when the container starts
CMD ["python", "inference.py"]
