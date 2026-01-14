FROM python:3.10-slim-buster

WORKDIR /app
COPY . /app

# Switch to archive repositories for Debian Buster
RUN sed -i 's/deb.debian.org/archive.debian.org/g' /etc/apt/sources.list && \
    sed -i 's|security.debian.org/debian-security|archive.debian.org/debian-security|g' /etc/apt/sources.list && \
    sed -i '/stretch-updates/d' /etc/apt/sources.list

RUN apt update -y && apt install awscli -y

RUN pip install -r requirements.txt

CMD ["python3", "app.py"]