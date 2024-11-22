# Supported Models

ServerlessLLM supports a plethora of language models from [Huggingface (HF) Transformers](https://huggingface.co/models). This page lists the models and model architectures currently supported by ServerlessLLM.

To test a model, simply add it to the `supported_models.json` inside `/ServerlessLLM/tests/inference_tests` and the Github Actions will automatically test whether not it is supported.

## Text-only Language Models

Architecture      |Models        |Example HF Models   |vLLM |Transformers
------------------|--------------|--------------------|-----|-------------
`OPTForCausalLM`  |OPT, OPT-IML  |`facebook/opt-6.7b` |✅   |✅


