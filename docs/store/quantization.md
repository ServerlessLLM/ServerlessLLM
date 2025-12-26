---
sidebar_position: 2
---

# Quantization

> Note: Quantization is currently experimental, especially on multi-GPU machines. You may encounter issues when using this feature in multi-GPU environments.

ServerlessLLM currently supports `bitsandbytes` quantization, which reduces model memory usage by converting weights to lower-precision data types. You can configure this by passing a `BitsAndBytesConfig` object when loading a model.

Available precisions include:
- `int8`
- `fp4`
- `nf4`

> Note: CPU offloading and dequantization is not currently supported.

## 8-bit Quantization (`int8`)

8-bit quantization halves the memory usage compared to 16-bit precision with minimal impact on model accuracy. It is a robust and recommended starting point for quantization.

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# Configure 8-bit quantization
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True
)

# Load the model with the config
model_8bit = AutoModelForCausalLM.from_pretrained(
    "facebook/opt-1.3b",
    quantization_config=quantization_config,
    device_map="auto",
)
```

## 4-bit Quantization (`fp4`)
FP4 (4-bit Floating Point) quantization offers more aggressive memory savings than 8-bit. It is a good option for running very large models on consumer-grade hardware.

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# Configure 4-bit FP4 quantization
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="fp4"
)

# Load the model with the config
model_fp4 = AutoModelForCausalLM.from_pretrained(
    "facebook/opt-1.3b",
    quantization_config=quantization_config,
    device_map="auto",
)
```

## 4-bit Quantization (`nf4`)
NF4 (4-bit NormalFloat) is an advanced data type optimized for models whose weights follow a normal distribution. NF4 is generally the recommended 4-bit option as it often yields better model accuracy compared to FP4.

```python
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# Configure 4-bit NF4 quantization
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4"
)

# Load the model with the config
model_nf4 = AutoModelForCausalLM.from_pretrained(
    "facebook/opt-1.3b",
    quantization_config=quantization_config,
    device_map="auto",
)
```

## `torch_dtype` (Data Type for Unquantized Layers)
The `torch_dtype` parameter sets the data type for model layers that are not quantized (e.g. `LayerNorm`). Setting this to `torch.float16` or `torch.bfloat16` can further reduce memory usage. If unspecified, these layers default to `torch.float16`.

```python
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# Configure 4-bit NF4 quantization
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4"
)

# Load model, casting non-quantized layers to float16
model_mixed_precision = AutoModelForCausalLM.from_pretrained(
    "facebook/opt-1.3b",
    quantization_config=quantization_config,
    torch_dtype=torch.float16,
    device_map="auto",
)
```

For further information, consult the [HuggingFace Documentation for BitsAndBytes](https://huggingface.co/docs/transformers/main/en/quantization/bitsandbytes).

