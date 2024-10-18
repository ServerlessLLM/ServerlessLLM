# How to test replay command

## Run the server

1. Set the `MODEL_FOLDER` environment variable to specify the folder to store the models:

    ```sh
    cd path/to/your-folder/
    mkdir models
    export MODEL_FOLDER=$PWD/models/
    ```

2. Run the server:

    ```sh
    bash examples/clean.sh
    bash examples/run.sh
    ```

## Run the test

In another terminal:

1. Configuration:

    Set the `LLM_SERVER_URL` environment variable to specify the server URL:

    ```sh
    export LLM_SERVER_URL=http://127.0.0.1:8343/
    ```

    Install the package:

    ```sh
    pip install .
    pip install datasets
    ```
    <!-- I want to add datasets to requirements.txt but it would cause unwanted errors(Segmentation fault). So, I will leave it as a manual step for now. -->

2. Run the test:

    Change directory to the test folder:

    ```sh
    cd tests/replay_test/
    ```

    Prepare the gsm8k dataset:

    ```sh
    python convert_gsm8k.py
    ```

    Run the test:
    ```sh
    python deploy_dummy_models.py --num_models 7
    python generate_random_workload.py --num_models 7 --request_rate 1 --duration_minutes 1
    sllm-cli replay --workload workload.json --dataset gsm8k_dataset.json
    ```
