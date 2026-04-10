FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install
COPY traffic_env/server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the full project
COPY . .

# Expose port (HF Spaces default = 7860)
EXPOSE 7860

# Run the FastAPI server via uvicorn
CMD ["uvicorn", "traffic_env.server.app:app", "--host", "0.0.0.0", "--port", "7860"]
