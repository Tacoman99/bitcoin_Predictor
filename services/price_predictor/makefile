SHELL:=/bin/bash
train:
	source ../setup_credentials.sh && \
	poetry run python src/training.py

predict:
	source ../setup_credentials.sh && \
	poetry run python src/predictor.py

api:
	source ../setup_credentials.sh && \
	poetry run python src/api.py

request:
	curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"product_id":"BTC/USD"}'

invalid-request:
	curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"product_id":"ETH/USD"}'

copy-tools:
	cp -r ../../tools .

build: copy-tools
	docker build -t price-predictor-api .

run: build
	docker run \
		--env-file ../credentials.env \
		-p 5000:80 \
		price-predictor-api
request-production:
	curl -X POST https://price-predictor-1-hirejorgecampos-cryptomlpredictor-production.deployments.quix.io/predict -H "Content-Type: application/json" -d '{"product_id":"BTC/USD"}