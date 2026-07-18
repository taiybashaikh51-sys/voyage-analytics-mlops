FROM python:3.10-slim

WORKDIR /app

# Copy API code and requirements
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/ .
COPY models/ /models/

EXPOSE 5000

CMD ["python", "app.py"]