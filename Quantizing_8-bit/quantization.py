from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_path = "./qwen-complaint-agent-merged"

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,           # Set to True for 8-bit quantization
    llm_int8_threshold=6.0,      # Default threshold, can be tuned
    llm_int8_skip_modules=None   # Quantize all modules
)

print("Loading model in 8-bit quantized mode...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    quantization_config=bnb_config
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Save quantized model (optional, for future direct loading)
model.save_pretrained("./qwen-complaint-agent-merged-8bit")
tokenizer.save_pretrained("./qwen-complaint-agent-merged-8bit")

print("Model quantized to 8-bit and saved!")
