# Use the Python 3.12 slim Bullseye image as the base
FROM python:3.11.5-alpine

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --upgrade pip &&\
    python -m pip install --upgrade build && pip install -r ./requirements.txt

# Copy the FastAPI application code
COPY ./app ./app

# Expose the port the app runs on (change if your app uses a different port)
EXPOSE 8000

# Start the FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]