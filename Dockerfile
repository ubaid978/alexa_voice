FROM python:3.10

WORKDIR /code

# 👇 Add destination path (.)
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]

