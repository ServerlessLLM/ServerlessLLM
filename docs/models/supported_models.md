# Supported Models

ServerlessLLM supports a plethora of language models from [Huggingface (HF) Transformers](https://huggingface.co/models). This page lists the models and model architectures currently supported by ServerlessLLM.

To test a model, simply add it to the `supported_models.json` inside `/ServerlessLLM/tests/inference_test` and the Github Actions will automatically test whether not it is supported.

## Text-only Language Models

| Architecture      | Models       | Example HF Models      | vLLM | SGLang | Transformers |
|-------------------|--------------|------------------------|------|--------|--------------|
| `OPTForCausalLM`  | OPT, OPT-IML | `facebook/opt-1.3b`    | ✅    | ✅      | ✅            |
| `Qwen2ForCausalLM`| Qwen2.5      | `Qwen/Qwen2.5-1.5B`    | ✅    | ✅      | ✅            |

## Vision Language Models

| Architecture                         | Models  | Example HF Models            | vLLM | SGLang | Transformers |
|--------------------------------------|---------|------------------------------|------|--------|--------------|
| `Qwen2VLForConditionalGeneration`    | Qwen2VL | `Qwen/Qwen2-VL-2B-Instruct`  | ✅    | ✅      | ✅            |
