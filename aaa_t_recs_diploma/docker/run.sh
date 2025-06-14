#!/bin/bash

set -e

if [ -z "$PROJECT_DIR" ] || [ -z "$CONTAINER_WORKDIR" ] || [ -z "$DOCKER_IMAGE" ]; then
  echo "Ошибка: Необходимо задать переменные окружения:"
  echo "  PROJECT_DIR, CONTAINER_WORKDIR, DOCKER_IMAGE"
  exit 1
fi

docker run -it \
  --env CONTAINER_WORKDIR=$CONTAINER_WORKDIR \
  --volume="$PROJECT_DIR/cmd:$CONTAINER_WORKDIR/cmd" \
  --volume="$PROJECT_DIR/config:$CONTAINER_WORKDIR/config" \
  --volume="$PROJECT_DIR/interval:$CONTAINER_WORKDIR/interval" \
  --volume="$PROJECT_DIR/views:$CONTAINER_WORKDIR/views" \
  --volume="$PROJECT_DIR/src:$CONTAINER_WORKDIR/src" \
  --volume="$PROJECT_DIR/data:$CONTAINER_WORKDIR/data" \
  --volume="$PROJECT_DIR/poetry.lock:$CONTAINER_WORKDIR/poetry.lock" \
  --volume="$PROJECT_DIR/pyproject.toml:$CONTAINER_WORKDIR/pyproject.toml" \
  "$DOCKER_IMAGE" bash
