FROM python:3.10

WORKDIR /app

ENV PYTHONPATH=/app
ENV PYTHONWARNINGS=ignore

RUN groupadd --gid 1000 myuser \
    && useradd -u 1000 --gid 1000 myuser \
    && chown -R myuser /app \
    && mkdir /home/myuser \
    && chown -R myuser /home/myuser

COPY requirements.txt .
RUN pip --no-cache-dir install -r requirements.txt

USER myuser

EXPOSE 8888