from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load the pre-trained GPT-2 model
model = GPT2LMHeadModel.from_pretrained('gpt2')
# Instatniate the GPT-2 tokenizer - responsible for encoding and decoding text 
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Generate text
input_prompt = "Once upon a time, there was a young woman named Alice who lived in a small town."
input_ids = tokenizer.encode(input_prompt, return_tensors="pt")
output = model.generate(input_ids, max_length=100)
generated_text = tokenizer.decode(output[0])

print(generated_text)