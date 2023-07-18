# Use an official Python runtime as the base image
FROM python:3.10.6

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file to the container
COPY requirements.txt .

# Install the project dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project files to the container
COPY . .

# Expose the port on which your Flask application is listening
EXPOSE 5000

# Set the command to run your Flask application
CMD ["python", "app.py"]
