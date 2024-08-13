FROM python:3.10-slim-buster
WORKDIR /app
COPY . /app

RUN apt update -y && apt insatll awscli -y

RUN pip insatll -r requirements.txt
CMD ["python3", "app.py"]