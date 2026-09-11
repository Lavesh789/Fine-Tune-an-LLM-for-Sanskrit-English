import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Page Configuration (Fixed title parameter)
st.set_page_config(
    page_title="Sanskrit-English AI Assistant",
    page_icon="🕉️",
    layout="centered"
)

st.title("🕉️ Sanskrit-English LLM Assistant")
st.markdown(
    "Fine-tuned **Llama-3.2-1B** for Sanskrit translation, verse explanations, and philosophical context."
)

# Model & Tokenizer Loader
@st.cache_resource(show_spinner="Loading fine-tuned model into memory...")
def load_model_and_tokenizer():
    base_model_name = "unsloth/Llama-3.2-1B-Instruct"
    adapter_path = "sanskrit_lora_model" # Local path or Hugging Face repo ID

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        device_map="auto" if device == "cuda" else None
    )

    try:
        model = PeftModel.from_pretrained(base_model, adapter_path)
    except Exception as e:
        st.warning(f"Could not load LoRA adapters from '{adapter_path}'. Falling back to base model.")
        model = base_model

    model.eval()
    return model, tokenizer, device

# Initialize Model
try:
    model, tokenizer, device = load_model_and_tokenizer()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error initializing model: {e}")
    st.stop()

# Sidebar Controls
st.sidebar.header("Generation Parameters")
task_type = st.sidebar.selectbox(
    "Select Task",
    ["Sanskrit to English Translation", "Explain Sanskrit Verse", "English to Sanskrit Translation"]
)
max_tokens = st.sidebar.slider("Max Output Tokens", min_value=32, max_value=512, value=128, step=32)
temperature = st.sidebar.slider("Temperature", min_value=0.1, max_value=1.0, value=0.3, step=0.1)

# Input Presets
default_inputs = {
    "Sanskrit to English Translation": "कर्मण्येवाधिकारस्ते मा फलेषु कदाचन।",
    "Explain Sanskrit Verse": "विद्या ददाति विनयं विनयाद्याति पात्रताम्।",
    "English to Sanskrit Translation": "Truth alone triumphs, not falsehood."
}

user_input = st.text_area(
    "Enter Sanskrit or English Text:",
    value=default_inputs.get(task_type, ""),
    height=100
)

# Prompt Formatting
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

# Inference Pipeline
if st.button("Generate Answer", type="primary"):
    if not user_input.strip():
        st.warning("Please enter some text first.")
    else:
        if task_type == "Sanskrit to English Translation":
            instruction = "Translate the following Sanskrit text to English."
        elif task_type == "Explain Sanskrit Verse":
            instruction = "Explain the underlying philosophy of this Sanskrit verse."
        else:
            instruction = "Translate the following English text to Sanskrit."

        prompt = alpaca_prompt.format(instruction, user_input, "")

        with st.spinner("Generating response..."):
            inputs = tokenizer([prompt], return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True if temperature > 0.1 else False,
                    pad_token_id=tokenizer.eos_token_id
                )

            full_response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            
            if "### Response:" in full_response:
                response = full_response.split("### Response:")[1].strip()
            else:
                response = full_response

        st.markdown("### Output")
        st.info(response)
