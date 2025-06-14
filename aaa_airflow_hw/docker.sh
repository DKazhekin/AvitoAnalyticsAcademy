#!/bin/bash

IMAGE_NAME=denisska390/airflow_homework

docker build -t $IMAGE_NAME .
docker push $IMAGE_NAME