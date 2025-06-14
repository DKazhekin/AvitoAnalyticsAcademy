#!/bin/bash

source $CONTAINER_WORKDIR/venv/bin/activate
cd cmd/app/
uvicorn main:fast_api_app --host $HOST --port $PORT --env-file ../../.env --reload
