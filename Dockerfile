FROM python:3.10-slim
ENV PYTHONBUFFERED True
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./
RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt
CMD ["python", "main.py"]