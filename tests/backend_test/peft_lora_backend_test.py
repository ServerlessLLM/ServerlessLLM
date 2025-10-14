# ---------------------------------------------------------------------------- #
#  ServerlessLLM                                                               #
#  Copyright (c) ServerlessLLM Team 2025                                       #
#                                                                              #
#  Licensed under the Apache License, Version 2.0 (the "License");             #
#  you may not use this file except in compliance with the License.            #
#                                                                              #
#  You may obtain a copy of the License at                                     #
#                                                                              #
#                  http://www.apache.org/licenses/LICENSE-2.0                  #
#                                                                              #
#  Unless required by applicable law or agreed to in writing, software         #
#  distributed under the License is distributed on an "AS IS" BASIS,           #
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.    #
#  See the License for the specific language governing permissions and         #
#  limitations under the License.                                              #
# ---------------------------------------------------------------------------- #
import os
from unittest.mock import MagicMock, patch

import pytest
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from sllm.ft_backends.backend_utils import FineTuningBackendStatus
from sllm.ft_backends.peft_lora_backend import (
    FineTuningStatus,
    PeftLoraBackend,
)


@pytest.fixture
def model_name():
    return "facebook/opt-125m"


@pytest.fixture
def backend_config():
    return {
        "pretrained_model_name_or_path": "facebook/opt-125m",
        "torch_dtype": "float16",
        "hf_model_class": "AutoModelForCausalLM",
        "device_map": "auto",
    }


@pytest.fixture
def peft_lora_backend(model_name, backend_config):
    yield PeftLoraBackend(model_name, backend_config)


@pytest.fixture
def mock_model():
    model = MagicMock(spec=AutoModelForCausalLM)
    model.config = MagicMock()
    model.config.vocab_size = 50272
    return model


@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock(spec=AutoTokenizer)
    tokenizer.pad_token = "<pad>"
    tokenizer.eos_token = "</s>"
    tokenizer.pad_token_id = 1
    tokenizer.eos_token_id = 2
    return tokenizer


@pytest.fixture
def mock_dataset():
    dataset = MagicMock(spec=Dataset)
    dataset.__len__ = MagicMock(return_value=100)
    return dataset


@pytest.fixture
def mock_peft_model():
    peft_model = MagicMock()
    peft_model.config = MagicMock()
    return peft_model


@pytest.fixture
def mock_trainer():
    trainer = MagicMock()
    trainer.train = MagicMock()
    return trainer


@pytest.fixture
def sample_request_data():
    return {
        "job_id": "test_job_123",
        "dataset_config": {
            "dataset_source": "hf_hub",
            "hf_dataset_name": "test_dataset",
            "tokenization_field": "text",
            "split": "train",
        },
        "lora_config": {
            "r": 16,
            "lora_alpha": 32,
            "target_modules": ["q_proj", "v_proj"],
            "lora_dropout": 0.1,
            "bias": "none",
            "task_type": "CAUSAL_LM",
        },
        "training_config": {
            "num_train_epochs": 1,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 1,
            "learning_rate": 2e-4,
            "warmup_steps": 100,
            "logging_steps": 10,
            "save_steps": 500,
            "save_total_limit": 2,
            "remove_unused_columns": False,
            "dataloader_pin_memory": False,
        },
        "output_dir": "test_lora_adapter",
    }


def test_init(peft_lora_backend, model_name, backend_config):
    assert peft_lora_backend.model_name == model_name
    assert peft_lora_backend.backend_config == backend_config
    assert peft_lora_backend.status == FineTuningBackendStatus.UNINITIALIZED
    assert peft_lora_backend.model is None
    assert peft_lora_backend.tokenizer is None
    assert (
        peft_lora_backend.pretrained_model_name_or_path == "facebook/opt-125m"
    )


@patch("sllm.ft_backends.peft_lora_backend.load_model")
@patch("sllm.ft_backends.peft_lora_backend.AutoTokenizer.from_pretrained")
def test_init_backend_success(
    mock_tokenizer_from_pretrained,
    mock_load_model,
    peft_lora_backend,
    mock_model,
    mock_tokenizer,
):
    mock_load_model.return_value = mock_model
    mock_tokenizer_from_pretrained.return_value = mock_tokenizer

    peft_lora_backend.init_backend()

    assert peft_lora_backend.status == FineTuningBackendStatus.RUNNING
    assert peft_lora_backend.model == mock_model
    assert peft_lora_backend.tokenizer == mock_tokenizer

    # Verify load_model was called with correct parameters
    mock_load_model.assert_called_once()
    call_args = mock_load_model.call_args
    assert call_args[0][0] == os.path.join(
        "./models", "transformers", "facebook/opt-125m"
    )
    assert call_args[1]["device_map"] == "auto"
    assert call_args[1]["torch_dtype"] == torch.float16
    assert call_args[1]["hf_model_class"] == "AutoModelForCausalLM"


@patch("sllm.ft_backends.peft_lora_backend.load_model")
@patch("sllm.ft_backends.peft_lora_backend.AutoTokenizer.from_pretrained")
def test_init_backend_with_local_tokenizer(
    mock_tokenizer_from_pretrained,
    mock_load_model,
    peft_lora_backend,
    mock_model,
    mock_tokenizer,
):
    mock_load_model.return_value = mock_model
    mock_tokenizer_from_pretrained.return_value = mock_tokenizer

    with patch("os.path.exists", return_value=True):
        peft_lora_backend.init_backend()

    mock_tokenizer_from_pretrained.assert_called_once_with(
        os.path.join(
            "./models", "transformers", "facebook/opt-125m", "tokenizer"
        )
    )


@patch("sllm.ft_backends.peft_lora_backend.load_model")
def test_init_backend_already_initialized(mock_load_model, peft_lora_backend):
    peft_lora_backend.status = FineTuningBackendStatus.RUNNING
    peft_lora_backend.init_backend()
    mock_load_model.assert_not_called()


def test_init_backend_invalid_torch_dtype(peft_lora_backend):
    peft_lora_backend.backend_config["torch_dtype"] = "invalid_dtype"

    with patch(
        "sllm.ft_backends.peft_lora_backend.load_model"
    ) as mock_load_model:
        mock_load_model.return_value = MagicMock()
        with patch(
            "sllm.ft_backends.peft_lora_backend.AutoTokenizer.from_pretrained"
        ) as mock_tokenizer:
            mock_tokenizer.return_value = MagicMock()
            peft_lora_backend.init_backend()

            # Should use torch.float16 as fallback
            call_args = mock_load_model.call_args
            assert call_args[1]["torch_dtype"] == torch.float16


def test_fine_tuning_status_init():
    status = FineTuningStatus("test_job")
    assert status.job_id == "test_job"
    assert status.state == FineTuningBackendStatus.PENDING
    assert status.logs == []
    assert status.metrics == {}
    assert status.created_at > 0
    assert status.updated_at == status.created_at


def test_fine_tuning_status_lifecycle():
    status = FineTuningStatus("test_job")

    # Test start
    status.start()
    assert status.state == FineTuningBackendStatus.RUNNING

    # Test update_metrics
    status.update_metrics(epoch=1, loss=0.5)
    assert status.metrics["epoch"] == 1
    assert status.metrics["loss"] == 0.5

    # Test log
    status.log("Test message")
    assert len(status.logs) == 1
    assert "Test message" in status.logs[0]

    # Test complete
    status.complete()
    assert status.state == FineTuningBackendStatus.COMPLETED


def test_fine_tuning_status_fail():
    status = FineTuningStatus("test_job")
    status.fail("Test error")
    assert status.state == FineTuningBackendStatus.FAILED
    assert any("Test error" in log for log in status.logs)


def test_fine_tuning_status_abort():
    status = FineTuningStatus("test_job")
    status.abort()
    assert status.state == FineTuningBackendStatus.ABORTED
    assert any("Job aborted by user" in log for log in status.logs)


def test_fine_tuning_status_get_status():
    status = FineTuningStatus("test_job")
    status.start()
    status.update_metrics(epoch=1, loss=0.5)
    status.log("Test message")

    result = status.get_status()
    assert result["job_id"] == "test_job"
    assert result["state"] == "running"
    assert result["metrics"]["epoch"] == 1
    assert result["metrics"]["loss"] == 0.5
    assert len(result["logs"]) == 1
    assert "created_at" in result
    assert "updated_at" in result


@patch("sllm.ft_backends.peft_lora_backend.load_model")
@patch("sllm.ft_backends.peft_lora_backend.AutoTokenizer.from_pretrained")
def test_fine_tuning_not_initialized(
    mock_tokenizer_from_pretrained,
    mock_load_model,
    peft_lora_backend,
    sample_request_data,
):
    result = peft_lora_backend.fine_tuning(sample_request_data)
    assert "error" in result
    assert "Model not initialized" in result["error"]


@patch("sllm.ft_backends.peft_lora_backend.load_model")
@patch("sllm.ft_backends.peft_lora_backend.AutoTokenizer.from_pretrained")
@patch("sllm.ft_backends.peft_lora_backend.load_dataset")
@patch("sllm.ft_backends.peft_lora_backend.LoraConfig")
@patch("sllm.ft_backends.peft_lora_backend.get_peft_model")
@patch("sllm.ft_backends.peft_lora_backend.TrainingArguments")
@patch("sllm.ft_backends.peft_lora_backend.Trainer")
@patch("sllm.ft_backends.peft_lora_backend.save_lora")
def test_fine_tuning_success(
    mock_save_lora,
    mock_trainer_cls,
    mock_training_args_cls,
    mock_get_peft_model,
    mock_lora_config_cls,
    mock_load_dataset,
    mock_tokenizer_from_pretrained,
    mock_load_model,
    peft_lora_backend,
    mock_model,
    mock_tokenizer,
    mock_dataset,
    mock_peft_model,
    mock_trainer,
    sample_request_data,
):
    mock_load_model.return_value = mock_model
    mock_tokenizer_from_pretrained.return_value = mock_tokenizer
    mock_load_dataset.return_value = mock_dataset
    mock_lora_config_cls.return_value = MagicMock()
    mock_get_peft_model.return_value = mock_peft_model
    mock_training_args_cls.return_value = MagicMock()
    mock_trainer_cls.return_value = mock_trainer
    mock_save_lora.return_value = None

    peft_lora_backend.init_backend()

    result = peft_lora_backend.fine_tuning(sample_request_data)

    assert "model" in result
    assert "lora_save_path" in result
    assert result["model"] == "facebook/opt-125m"
    assert "test_lora_adapter" in result["lora_save_path"]

    assert (
        peft_lora_backend.ft_status.state == FineTuningBackendStatus.COMPLETED
    )


@patch("sllm.ft_backends.peft_lora_backend.load_model")
@patch("sllm.ft_backends.peft_lora_backend.AutoTokenizer.from_pretrained")
@patch("sllm.ft_backends.peft_lora_backend.load_dataset")
def test_fine_tuning_dataset_load_failure(
    mock_load_dataset,
    mock_tokenizer_from_pretrained,
    mock_load_model,
    peft_lora_backend,
    mock_model,
    mock_tokenizer,
    sample_request_data,
):
    mock_load_model.return_value = mock_model
    mock_tokenizer_from_pretrained.return_value = mock_tokenizer
    mock_load_dataset.side_effect = Exception("Dataset load failed")

    peft_lora_backend.init_backend()
    result = peft_lora_backend.fine_tuning(sample_request_data)

    assert "error" in result
    assert "Dataset load failed" in result["error"]
    assert peft_lora_backend.ft_status.state == FineTuningBackendStatus.FAILED


@patch("sllm.ft_backends.peft_lora_backend.load_model")
@patch("sllm.ft_backends.peft_lora_backend.AutoTokenizer.from_pretrained")
@patch("sllm.ft_backends.peft_lora_backend.load_dataset")
@patch("sllm.ft_backends.peft_lora_backend.LoraConfig")
def test_fine_tuning_lora_config_failure(
    mock_lora_config_cls,
    mock_load_dataset,
    mock_tokenizer_from_pretrained,
    mock_load_model,
    peft_lora_backend,
    mock_model,
    mock_tokenizer,
    mock_dataset,
    sample_request_data,
):
    mock_load_model.return_value = mock_model
    mock_tokenizer_from_pretrained.return_value = mock_tokenizer
    mock_load_dataset.return_value = mock_dataset
    mock_lora_config_cls.side_effect = Exception("Invalid LoRA config")

    peft_lora_backend.init_backend()
    result = peft_lora_backend.fine_tuning(sample_request_data)

    assert "error" in result
    assert "Invalid LoRA config" in result["error"]
    assert peft_lora_backend.ft_status.state == FineTuningBackendStatus.FAILED


@patch("sllm.ft_backends.peft_lora_backend.load_model")
@patch("sllm.ft_backends.peft_lora_backend.AutoTokenizer.from_pretrained")
@patch("sllm.ft_backends.peft_lora_backend.load_dataset")
@patch("sllm.ft_backends.peft_lora_backend.LoraConfig")
@patch("sllm.ft_backends.peft_lora_backend.get_peft_model")
def test_fine_tuning_peft_model_failure(
    mock_get_peft_model,
    mock_lora_config_cls,
    mock_load_dataset,
    mock_tokenizer_from_pretrained,
    mock_load_model,
    peft_lora_backend,
    mock_model,
    mock_tokenizer,
    mock_dataset,
    sample_request_data,
):
    mock_load_model.return_value = mock_model
    mock_tokenizer_from_pretrained.return_value = mock_tokenizer
    mock_load_dataset.return_value = mock_dataset
    mock_lora_config_cls.return_value = MagicMock()
    mock_get_peft_model.side_effect = Exception("PEFT model creation failed")

    peft_lora_backend.init_backend()
    result = peft_lora_backend.fine_tuning(sample_request_data)

    assert "error" in result
    assert "PEFT model creation failed" in result["error"]
    assert peft_lora_backend.ft_status.state == FineTuningBackendStatus.FAILED


@patch("sllm.ft_backends.peft_lora_backend.load_model")
@patch("sllm.ft_backends.peft_lora_backend.AutoTokenizer.from_pretrained")
@patch("sllm.ft_backends.peft_lora_backend.load_dataset")
@patch("sllm.ft_backends.peft_lora_backend.LoraConfig")
@patch("sllm.ft_backends.peft_lora_backend.get_peft_model")
@patch("sllm.ft_backends.peft_lora_backend.TrainingArguments")
def test_fine_tuning_training_args_failure(
    mock_training_args_cls,
    mock_get_peft_model,
    mock_lora_config_cls,
    mock_load_dataset,
    mock_tokenizer_from_pretrained,
    mock_load_model,
    peft_lora_backend,
    mock_model,
    mock_tokenizer,
    mock_dataset,
    mock_peft_model,
    sample_request_data,
):
    mock_load_model.return_value = mock_model
    mock_tokenizer_from_pretrained.return_value = mock_tokenizer
    mock_load_dataset.return_value = mock_dataset
    mock_lora_config_cls.return_value = MagicMock()
    mock_get_peft_model.return_value = mock_peft_model
    mock_training_args_cls.side_effect = Exception("Invalid training args")

    peft_lora_backend.init_backend()
    result = peft_lora_backend.fine_tuning(sample_request_data)

    assert "error" in result
    assert "Invalid training args" in result["error"]
    assert peft_lora_backend.ft_status.state == FineTuningBackendStatus.FAILED


@patch("sllm.ft_backends.peft_lora_backend.load_model")
@patch("sllm.ft_backends.peft_lora_backend.AutoTokenizer.from_pretrained")
@patch("sllm.ft_backends.peft_lora_backend.load_dataset")
@patch("sllm.ft_backends.peft_lora_backend.LoraConfig")
@patch("sllm.ft_backends.peft_lora_backend.get_peft_model")
@patch("sllm.ft_backends.peft_lora_backend.TrainingArguments")
@patch("sllm.ft_backends.peft_lora_backend.Trainer")
def test_fine_tuning_trainer_failure(
    mock_trainer_cls,
    mock_training_args_cls,
    mock_get_peft_model,
    mock_lora_config_cls,
    mock_load_dataset,
    mock_tokenizer_from_pretrained,
    mock_load_model,
    peft_lora_backend,
    mock_model,
    mock_tokenizer,
    mock_dataset,
    mock_peft_model,
    sample_request_data,
):
    mock_load_model.return_value = mock_model
    mock_tokenizer_from_pretrained.return_value = mock_tokenizer
    mock_load_dataset.return_value = mock_dataset
    mock_lora_config_cls.return_value = MagicMock()
    mock_get_peft_model.return_value = mock_peft_model
    mock_training_args_cls.return_value = MagicMock()
    mock_trainer_cls.side_effect = Exception("Trainer creation failed")

    peft_lora_backend.init_backend()
    result = peft_lora_backend.fine_tuning(sample_request_data)

    assert "error" in result
    assert "Trainer creation failed" in result["error"]
    assert peft_lora_backend.ft_status.state == FineTuningBackendStatus.FAILED


@patch("sllm.ft_backends.peft_lora_backend.load_model")
@patch("sllm.ft_backends.peft_lora_backend.AutoTokenizer.from_pretrained")
@patch("sllm.ft_backends.peft_lora_backend.load_dataset")
@patch("sllm.ft_backends.peft_lora_backend.LoraConfig")
@patch("sllm.ft_backends.peft_lora_backend.get_peft_model")
@patch("sllm.ft_backends.peft_lora_backend.TrainingArguments")
@patch("sllm.ft_backends.peft_lora_backend.Trainer")
@patch("sllm.ft_backends.peft_lora_backend.save_lora")
def test_fine_tuning_training_failure(
    mock_save_lora,
    mock_trainer_cls,
    mock_training_args_cls,
    mock_get_peft_model,
    mock_lora_config_cls,
    mock_load_dataset,
    mock_tokenizer_from_pretrained,
    mock_load_model,
    peft_lora_backend,
    mock_model,
    mock_tokenizer,
    mock_dataset,
    mock_peft_model,
    mock_trainer,
    sample_request_data,
):
    mock_load_model.return_value = mock_model
    mock_tokenizer_from_pretrained.return_value = mock_tokenizer
    mock_load_dataset.return_value = mock_dataset
    mock_lora_config_cls.return_value = MagicMock()
    mock_get_peft_model.return_value = mock_peft_model
    mock_training_args_cls.return_value = MagicMock()
    mock_trainer_cls.return_value = mock_trainer
    mock_trainer.train.side_effect = Exception("Training failed")

    peft_lora_backend.init_backend()
    result = peft_lora_backend.fine_tuning(sample_request_data)

    assert "error" in result
    assert "Training failed" in result["error"]
    assert peft_lora_backend.ft_status.state == FineTuningBackendStatus.FAILED


def test_load_dataset_hf_hub(peft_lora_backend, mock_tokenizer):
    dataset_config = {
        "dataset_source": "hf_hub",
        "hf_dataset_name": "test_dataset",
        "tokenization_field": "text",
        "split": "train",
    }

    with patch(
        "sllm.ft_backends.peft_lora_backend.load_dataset"
    ) as mock_load_dataset:
        mock_dataset = MagicMock()
        mock_load_dataset.return_value = mock_dataset
        mock_dataset.map.return_value = mock_dataset

        result = peft_lora_backend._load_dataset(dataset_config, mock_tokenizer)

        mock_load_dataset.assert_called_once_with("test_dataset", split="train")
        mock_dataset.map.assert_called_once()


def test_load_dataset_local(peft_lora_backend, mock_tokenizer):
    dataset_config = {
        "dataset_source": "local",
        "data_files": "test.json",
        "tokenization_field": "text",
        "extension_type": "json",
        "split": "train",
    }

    with patch(
        "sllm.ft_backends.peft_lora_backend.load_dataset"
    ) as mock_load_dataset:
        mock_dataset = MagicMock()
        mock_load_dataset.return_value = mock_dataset
        mock_dataset.map.return_value = mock_dataset

        result = peft_lora_backend._load_dataset(dataset_config, mock_tokenizer)

        mock_load_dataset.assert_called_once_with(
            "json", data_files="test.json", split="train"
        )
        mock_dataset.map.assert_called_once()


def test_load_dataset_invalid_source(peft_lora_backend, mock_tokenizer):
    dataset_config = {
        "dataset_source": "invalid",
        "hf_dataset_name": "test_dataset",
        "tokenization_field": "text",
    }

    with pytest.raises(
        ValueError,
        match="Invalid 'dataset_source'. Must be 'hf_hub' or 'local'.",
    ):
        peft_lora_backend._load_dataset(dataset_config, mock_tokenizer)


def test_load_dataset_missing_hf_dataset_name(
    peft_lora_backend, mock_tokenizer
):
    dataset_config = {
        "dataset_source": "hf_hub",
        "tokenization_field": "text",
    }

    with pytest.raises(ValueError, match="hf_dataset_name must be provided"):
        peft_lora_backend._load_dataset(dataset_config, mock_tokenizer)


def test_load_dataset_missing_extension_type(peft_lora_backend, mock_tokenizer):
    dataset_config = {
        "dataset_source": "local",
        "data_files": "test.json",
        "tokenization_field": "text",
    }

    with pytest.raises(ValueError, match="extension_type must be provided"):
        peft_lora_backend._load_dataset(dataset_config, mock_tokenizer)


def test_load_dataset_missing_data_files(peft_lora_backend, mock_tokenizer):
    dataset_config = {
        "dataset_source": "local",
        "extension_type": "json",
        "tokenization_field": "text",
    }

    with pytest.raises(ValueError, match="data_files must be provided"):
        peft_lora_backend._load_dataset(dataset_config, mock_tokenizer)


@patch("sllm.ft_backends.peft_lora_backend.load_model")
@patch("sllm.ft_backends.peft_lora_backend.AutoTokenizer.from_pretrained")
def test_shutdown(
    mock_tokenizer_from_pretrained, mock_load_model, peft_lora_backend
):
    mock_load_model.return_value = MagicMock()
    mock_tokenizer_from_pretrained.return_value = MagicMock()

    peft_lora_backend.init_backend()

    assert peft_lora_backend.model is not None

    peft_lora_backend.shutdown()

    assert peft_lora_backend.status == FineTuningBackendStatus.DELETING
    assert not hasattr(peft_lora_backend, "model")


def test_shutdown_already_deleting(peft_lora_backend):
    peft_lora_backend.status = FineTuningBackendStatus.DELETING
    peft_lora_backend.shutdown()
    assert peft_lora_backend.status == FineTuningBackendStatus.DELETING


@patch("sllm.ft_backends.peft_lora_backend.load_model")
@patch("sllm.ft_backends.peft_lora_backend.AutoTokenizer.from_pretrained")
def test_stop(
    mock_tokenizer_from_pretrained, mock_load_model, peft_lora_backend
):
    mock_load_model.return_value = MagicMock()
    mock_tokenizer_from_pretrained.return_value = MagicMock()

    peft_lora_backend.init_backend()
    peft_lora_backend.stop()

    assert peft_lora_backend.status == FineTuningBackendStatus.DELETING


def test_stop_already_stopping(peft_lora_backend):
    peft_lora_backend.status = FineTuningBackendStatus.STOPPING
    peft_lora_backend.stop()
    assert peft_lora_backend.status == FineTuningBackendStatus.STOPPING


def test_stop_already_deleting(peft_lora_backend):
    peft_lora_backend.status = FineTuningBackendStatus.DELETING
    peft_lora_backend.stop()
    assert peft_lora_backend.status == FineTuningBackendStatus.DELETING


def test_fine_tuning_with_custom_job_id(peft_lora_backend, sample_request_data):
    custom_job_id = "custom_job_456"
    sample_request_data["job_id"] = custom_job_id

    peft_lora_backend.status = FineTuningBackendStatus.RUNNING
    peft_lora_backend.model = MagicMock()
    peft_lora_backend.tokenizer = MagicMock()

    with patch(
        "sllm.ft_backends.peft_lora_backend.load_dataset"
    ) as mock_load_dataset:
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=100)
        mock_load_dataset.return_value = mock_dataset

        with patch(
            "sllm.ft_backends.peft_lora_backend.LoraConfig"
        ) as mock_lora_config:
            mock_lora_config.return_value = MagicMock()

            with patch(
                "sllm.ft_backends.peft_lora_backend.get_peft_model"
            ) as mock_get_peft:
                mock_get_peft.return_value = MagicMock()

                with patch(
                    "sllm.ft_backends.peft_lora_backend.TrainingArguments"
                ) as mock_training_args:
                    mock_training_args.return_value = MagicMock()

                    with patch(
                        "sllm.ft_backends.peft_lora_backend.Trainer"
                    ) as mock_trainer_cls:
                        mock_trainer = MagicMock()
                        mock_trainer_cls.return_value = mock_trainer

                        with patch(
                            "sllm.ft_backends.peft_lora_backend.save_lora"
                        ):
                            peft_lora_backend.fine_tuning(sample_request_data)

                            assert (
                                peft_lora_backend.ft_status.job_id
                                == custom_job_id
                            )
