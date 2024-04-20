import os
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_name = "gpt2"

# Get the directory of the current script
script_dir = os.path.dirname(os.path.realpath(__file__))

# Define the directory containing the model files relative to the script directory
model_directory = os.path.join(script_dir, "fine-tuned-gpt2")
# Get the list of files in the model directory
model_files = os.listdir(model_directory)

# Check if the directory contains the necessary files
required_files = ["config.json", "generation_config.json"]
if not all(file in model_files for file in required_files):
    raise FileNotFoundError("The model directory does not contain all the required files.")

# Load the fine-tuned GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(model_directory, local_files_only=True)

# Set the pad token ID to the EOS token ID
tokenizer.pad_token_id = tokenizer.eos_token_id

model = GPT2LMHeadModel.from_pretrained(model_directory, local_files_only=True)

# Define a function to generate responses
def generate_response(prompt, generated_responses):
    response = ""
    while len(response) == 0 or response in generated_responses:  # Loop until a valid response is obtained
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        attention_mask = torch.ones_like(input_ids)  # Set attention mask to 1 for all tokens
        output = model.generate(input_ids, attention_mask=attention_mask, max_length=100, num_return_sequences=1)
        response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# Interactive loop to ask questions and get answers
generated_responses = set()  # Set to store generated responses
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Goodbye!")
        break
    response = generate_response(user_input, generated_responses)
    generated_responses.add(response)
    print("Bot:", response)



