FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -r hackerix/requirements.txt
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
