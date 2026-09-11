# Sanskrit-English LLM Instruction Fine-Tuning

An end-to-end, lightweight machine learning solution designed to instruction fine-tune open-source Large Language Models (LLMs)
for multilingual Sanskrit-English NLP tasks. 

Built on top of Meta's Llama-3.2-1B-Instruct using QLoRA (4-bit quantization), 
this system addresses low-resource Indic language challenges such as Devanagari subword fragmentation, complex morphology, 
verse translation, and contextual explanation.

# Installation
pip install -r requirements.txt

#Tech Stack & Model Specs

Base Model: `unsloth/Llama-3.2-1B-Instruct`
Quantization: 4-bit NormalFloat (NF4) via `bitsandbytes`
Fine-Tuning Method: QLoRA (Rank $r=16$, $\alpha=16$) target modules applied to all linear projections
Prompt Format: Structured Alpaca / ChatML Instruction Schema
Deployment: Streamlit Cloud / Hugging Face Spaces interface
