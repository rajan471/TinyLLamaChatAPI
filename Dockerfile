# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set working directory in the container
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Download the Hugging Face model
RUN python -c "from transformers import pipeline; pipeline('text-generation', model='TinyLlama/TinyLlama-1.1B-Chat-v1.0')"

# Make port 5000 available to the world outside this container
EXPOSE 5001

# Run the application
CMD ["python", "app.py"] 