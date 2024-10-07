# Use an official Python runtime as a parent image
FROM python:3.10

# Set the working directory
WORKDIR /app

# Update and install minimal dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    mesa-utils \
    libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set environment variable for numba cache directory
ENV NUMBA_CACHE_DIR=/tmp/numba_cache

# Create the numba cache directory
RUN mkdir -p /tmp/numba_cache

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY main.py .
COPY circulares_info_extraction/ circulares_info_extraction/

# Copy the environment file
COPY .env .

# Expose the port if your application uses one (e.g., if you're running a web server)
# EXPOSE 8000  # Uncomment if your application listens on a specific port

# Define the command to run your application (replace this with your actual entrypoint)
CMD ["python", "main.py"]
