FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /

RUN apt-get update && apt-get install -y python3-venv

COPY . /app/

RUN python3 -m venv /opt/venv && \
    . /opt/venv/bin/activate && \
    pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD ["/opt/venv/bin/python", "app.py"]
