# ---------------------------------------------------------------------------- #
#  serverlessllm                                                               #
#  copyright (c) serverlessllm team 2024                                       #
#                                                                              #
#  licensed under the apache license, version 2.0 (the "license");             #
#  you may not use this file except in compliance with the license.            #
#                                                                              #
#  you may obtain a copy of the license at                                     #
#                                                                              #
#                  http://www.apache.org/licenses/license-2.0                  #
#                                                                              #
#  unless required by applicable law or agreed to in writing, software         #
#  distributed under the license is distributed on an "as is" basis,           #
#  without warranties or conditions of any kind, either express or implied.    #
#  see the license for the specific language governing permissions and         #
#  limitations under the license.                                              #
# ---------------------------------------------------------------------------- #
import peft 
import transformers
from transformers import TrainingArguments, Trainer 
from peft import LoraConfig, get_peft_model, PeftModel 
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from sllm.serve.logger import init_logger
# from sllm_store.transformers import save_lora

logger = init_logger(__name__)

class PeftBackend:
    def __init__(self, model_name: str, config: Optional[Dict[str, Any]] = None):
        self.model_name = model_name
        self.config = config

    async def fine_tune(self, fine_tune_params: dict):
        logger.info(f"Starting fine-tuning for model {self.model_name} with params: {fine_tune_params}")
        
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        foundation_model = AutoModelForCausalLM.from_pretrained(self.model_name)

        dataset_path = fine_tune_params.get("dataset")
        dataset = load_dataset(dataset_path)

        # for test, use sample data. delete later
        data = dataset.map(lambda samples: tokenizer(samples["prompt"]), batched=True)
        train_sample = data["train"].select(range(50))

        train_sample = train_sample.remove_columns('act')

        lora_config = fine_tune_params.get("lora_config")

        lora_config = LoraConfig(**lora_config_data)

        epochs = fine_tune_params.get("epochs", 1)
        learning_rate = fine_tune_params.get("learning_rate", 0.001)
        batch_size = fine_tune_params.get("batch_size", 32)

        peft_model = get_peft_model(foundation_model, lora_config)

        training_args = TrainingArguments(
            output_dir=output_directory,
            auto_find_batch_size=True, # Find a correct batch size that fits the size of Data.
            learning_rate=learning_rate,
            num_train_epochs=epochs,
            use_cpu=True
        )

        trainer = Trainer(
            model=peft_model,
            args=training_args,
            train_dataset=train_sample,
            data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
        )
        trainer.train()

        # save the model, use save_lora(), in sllm_store/transformers.py
        save_path = fine_tune_params.get("output_dir", "./saved_lora_model")
        # save_lora(peft_model, save_path)
        logger.info(f"Fine-tuning completed. LoRA model saved to {save_path}")

