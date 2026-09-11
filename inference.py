import torch
from unsloth import FastLanguageModel

def run_inference():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "sanskrit_lora_model",
        max_seq_length = 2048,
        dtype = None,
        load_in_4bit = True,
    )
    FastLanguageModel.for_inference(model)

    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

    prompt = alpaca_prompt.format(
        "Translate the following Sanskrit verse to English.",
        "सत्यमेव जयते नानृतम्।",
        ""
    )

    inputs = tokenizer([prompt], return_tensors = "pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens = 128, use_cache = True)
    print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])

if __name__ == "__main__":
    run_inference()
