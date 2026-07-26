FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements file first
COPY research_ai_backend/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir setuptools wheel && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

CMD ["uvicorn", "research_ai_backend.main:app", "--host", "0.0.0.0", "--port", "10000"]