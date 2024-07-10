docker build . -t serverlessllm/sllm-serve
docker build -f Dockerfile.worker . -t serverlessllm/sllm-serve-worker
