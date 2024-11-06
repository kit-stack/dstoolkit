# Use a Python base image
FROM python:3.8-slim

# Install dependencies needed to build psycopg2 from source
RUN apt-get update && apt-get install -y \
    libpq-dev gcc

# Set the working directory inside the container
WORKDIR /app

# Copy the project files into the /app directory
COPY . /app

# Set the PYTHONPATH to include /app
ENV PYTHONPATH="/app"

# Install Python dependencies
RUN pip install -r requirements.txt

# Run the predict_model.py script
CMD ["python", "src/models/predict_model.py"]