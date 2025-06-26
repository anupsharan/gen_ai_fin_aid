# Use a slim Python base image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code
COPY . .

# Expose the port Uvicorn will listen on (default is 8000)
EXPOSE 8000

# Command to run Uvicorn. 
# --host 0.0.0.0 is crucial for accessibility within Docker.
# --reload is for development and should be removed for production.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]