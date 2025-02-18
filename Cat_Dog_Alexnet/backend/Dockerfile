# Use an official Python runtime as a parent image
FROM python:3.9

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt file to the container (for caching purposes)
# Copy it to the /app/backend directory in the container
COPY requirements.txt /app/backend/requirements.txt

# Install the dependencies
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# Now, copy all remaining backend files (like app.py, model.py, etc.) into the container
COPY . /app/backend

# Expose the port FastAPI will run on
EXPOSE 8000

# Start FastAPI using Uvicorn (from the /app/backend folder)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
