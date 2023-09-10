from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import time

model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")

def generate_response(prompt, max_length=100, temperature=1.0):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    output = model.generate(input_ids, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2, temperature=temperature)

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

while True:
    user_input = input("Pertanyaan: ")
    if user_input.lower() == "exit":
        break

    response = generate_response(user_input, max_length=100, temperature=0)

    for char in response:
        print(char, end='', flush=True)
        time.sleep(0.02)
    print()

model.close()

# ChatGPT
# Fahri developer
