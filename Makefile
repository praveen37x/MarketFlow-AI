.PHONY: help install test seed benchmark run-api run-dashboard docker-up docker-down clean

help:
	@echo "MarketFlow-AI Developer Commands:"
	@echo "  make install        Install Python dependencies"
	@echo "  make test           Run all pytest unit and integration suites"
	@echo "  make seed           Populate Lakehouse tiers with historical data"
	@echo "  make benchmark      Execute performance & throughput benchmark"
	@echo "  make run-api        Start FastAPI production REST service"
	@echo "  make run-dashboard  Start interactive Streamlit dashboard"
	@echo "  make docker-up      Spin up entire stack in Docker containers"
	@echo "  make docker-down    Stop Docker containers"
	@echo "  make clean          Clean temporary caches and logs"

install:
	pip install -r requirements.txt

test:
	python -m pytest tests -v

seed:
	python scripts/seed_data.py

benchmark:
	python scripts/benchmark.py

run-api:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

run-dashboard:
	streamlit run src/dashboard/app.py

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
