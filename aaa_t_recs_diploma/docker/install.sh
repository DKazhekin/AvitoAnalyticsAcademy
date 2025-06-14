#!/bin/bash
python3 -m venv $1
source $1/bin/activate

pip install --upgrade pip setuptools wheel
pip install poetry

poetry update
poetry install --no-root
