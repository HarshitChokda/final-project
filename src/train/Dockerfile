FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY ../../requirements.txt requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code and data into the container
COPY ../../data/processed/ data/processed/
COPY . .

# Run the training script
CMD ["python", "train.py"]
