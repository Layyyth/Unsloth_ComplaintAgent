from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# === CONFIGURATION ===
# Path to your base model (can be a Hugging Face hub model or a local directory)
BASE_MODEL_PATH = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"  # or your local base model path

# Path to your trained LoRA adapter (output_dir from Unsloth/PEFT training)
LORA_ADAPTER_PATH = "./qwen-complaint-agent"

# Path to save the merged model
MERGED_MODEL_PATH = "./qwen-complaint-agent-merged"

# === LOAD BASE MODEL AND TOKENIZER ===
print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.bfloat16,   # or torch.float16 if you used fp16
    device_map="auto"
)
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)

# === LOAD LORA ADAPTER ===
print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)

# === MERGE THE ADAPTER ===
print("Merging LoRA adapter into base model...")
model = model.merge_and_unload()  # This returns a standard Hugging Face model

# === SAVE THE MERGED MODEL AND TOKENIZER ===
print(f"Saving merged model to {MERGED_MODEL_PATH} ...")
model.save_pretrained(MERGED_MODEL_PATH)
tokenizer.save_pretrained(MERGED_MODEL_PATH)

print("✅ Merge complete. You can now use the merged model for inference or deployment.")
