FROM python:3.8

COPY /requirements.txt .
COPY /server.py .

RUN pip install -r requirements.txt

CMD ["python3", "server.py"]