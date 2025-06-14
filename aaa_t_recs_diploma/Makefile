include .env
export

.PHONY: install run build up

install:
	@echo "Создание виртуального окружения Python .venv"
	@python3 -m venv .venv
	@echo "Установка зависимостей через poetry"
	@bash -c "source .venv/bin/activate && pip install poetry && poetry lock && poetry install --no-root"

install_dev: install
	@bash -c "source .venv/bin/activate && pip install -r requirements-dev.txt"
	@bash -c "source .venv/bin/activate && pre-commit install"

run:
	@echo "Запуск сервиса..."
	@bash -c "cd cmd/app && uvicorn main:fast_api_app --env-file ../../.env --reload"

build:
	@echo "Сборка Docker image..."
	@./docker/build.sh

up:
	@echo "Запуск Docker compose..."
	@bash -c "cd docker/ && docker compose up"

bash_run:
	@echo "Запуск Docker image..."
	@./docker/run.sh
