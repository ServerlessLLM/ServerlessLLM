# Supported Models

ServerlessLLM supports a plethora of language models from [Huggingface (HF) Transformers](https://huggingface.co/models). This page lists the models and model architectures currently supported by ServerlessLLM.

For other models, you can check the `config.json` file inside the model repository. If the `"architectures"` field contains a model architecture listed below, then it should be supported in theory.


## Text-only Language Models 

Architecture      |Models        |Example HF Models   |vLLM |Transformers |ONNX |TensorRT
------------------|--------------|--------------------|-----|-------------|-----|--------
`OPTForCausalLM`  |OPT, OPT-IML  |`facebook/opt-6.7b` |✅   |             |     |


## Multimodal Language Models 

Architecture      |Models        |Inputs | Example HF Models   |vLLM |Transformers |ONNX |TensorRT
------------------|--------------|-------|---------------------|-----|-------------|-----|--------
                  |              |       |                     |     |             |     |

# Model Support Policy
Yes.
