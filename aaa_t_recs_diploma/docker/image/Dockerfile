FROM python:3.11-slim

ENV CONTAINER_WORKDIR /app
ENV VENV_PATH="${CONTAINER_WORKDIR}/venv"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install poetry

WORKDIR ${CONTAINER_WORKDIR}

COPY ./docker/install.sh /tmp/install.sh
COPY ./poetry.lock .
COPY ./pyproject.toml .
COPY .env .
RUN chmod +x /tmp/install.sh
RUN /tmp/install.sh ${VENV_PATH}

COPY ./docker/run_app.sh .
RUN chmod +x ./run_app.sh
