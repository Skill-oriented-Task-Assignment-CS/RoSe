# Use an official Python runtime as the base image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /RoSe

# Copy the current directory contents into the container at /app
COPY . /RoSe

# Set the PYTHONPATH to include the /app/lib directory
ENV PYTHONPATH /RoSE/lib:$PYTHONPATH

EXPOSE 5000


# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Run your script when the container launches
CMD ["python", "app.py"]

