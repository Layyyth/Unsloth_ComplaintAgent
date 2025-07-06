from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the merged model and tokenizer
model = AutoModelForCausalLM.from_pretrained("./qwen-complaint-agent-merged")
tokenizer = AutoTokenizer.from_pretrained("./qwen-complaint-agent-merged")

# Push to the Hugging Face Hub
model.push_to_hub("LaythAbuJafar/QwenInstruct7b_ComplaintAgent_Unsloth")
tokenizer.push_to_hub("LaythAbuJafar/QwenInstruct7b_ComplaintAgent_Unsloth")
