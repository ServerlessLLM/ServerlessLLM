#!/bin/bash

cd examples/docker
docker compose down

conda deactivate

rm -f test/sinference_test/failed_models.json
