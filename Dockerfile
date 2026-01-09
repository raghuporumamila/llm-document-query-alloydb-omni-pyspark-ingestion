# Use a lightweight Python image
FROM python:3.9-slim

# Install Java (Spark requirement)
RUN apt-get update && apt-get install -y default-jdk-headless && apt-get clean

COPY requirements.txt .
# 4. Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set working directory
WORKDIR /app


# Copy your script into the container
COPY pyspark/ingest_pdf_alloydb.py .
COPY jars/postgresql-42.7.7.jar .
COPY docs docs
# Run the script when the container starts
CMD ["python", "ingest_pdf_alloydb.py"]