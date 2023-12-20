FROM python:3.10-slim
ENV PYTHONBUFFERED True
ENV APP_HOME /app
WORKDIR $APP_HOME
RUN apk update
RUN apk add --no-cache wget
RUN mkdir -p /credentials
RUN wget -O /credentials/sa-gcs.json https://storage.googleapis.com/bangkitcapstone-bloomy-bucket/service-account/sa-gcs.json
COPY . ./
RUN pip install -r requirements.txt
EXPOSE 8080
CMD ["python", "main.py"]

