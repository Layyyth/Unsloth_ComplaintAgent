from transformers import AutoModelForCausalLM, AutoTokenizer

# Path to your locally saved quantized model
quantized_model_path = "./qwen-complaint-agent-merged-8bit"

# Your target Hugging Face repo
hub_repo_id = "LaythAbuJafar/QwenInstruct7b_ComplaintAgent_8bit"

# Load the quantized model and tokenizer from local directory
print("Loading quantized model and tokenizer...")
model = AutoModelForCausalLM.from_pretrained(quantized_model_path)
tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)

# Push model and tokenizer to Hugging Face Hub
print(f"Pushing model to the Hugging Face Hub repo: {hub_repo_id} ...")
model.push_to_hub(hub_repo_id)
tokenizer.push_to_hub(hub_repo_id)

print("Quantized model and tokenizer successfully pushed!")
