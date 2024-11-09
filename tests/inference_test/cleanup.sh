#!/bin/bash

cd examples/docker
docker compose down

conda deactivate

rm -f tests/inference_test/failed_models.json
