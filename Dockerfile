FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Install the package
RUN pip install -e .

# Run the server
CMD exec uvicorn langgraph.server:app --host ${HOST:-0.0.0.0} --port ${PORT:-8080}