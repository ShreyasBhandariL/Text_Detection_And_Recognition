# Base image
FROM python:3.12

# Set the working directory
WORKDIR /usr/src/app

# Upgrade pip
RUN pip install --upgrade pip

# Copy requirements.txt
COPY requirements.txt .

# Install dependencies with timeout
RUN pip install --no-cache-dir --timeout=120 -r requirements.txt

# Copy the rest of the application code
COPY . .

# Set the environment variable for Flask
ENV FLASK_APP=app.py

# Expose port 5000
EXPOSE 5000

# Run the Flask application
CMD ["flask", "run", "--host=0.0.0.0"]
