import streamlit as st
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# st.set_page_config(page_title="Foreign Language Story Generator", layout="centered")

def generate_story(input_prompt: str, model, tokenizer):

    input_ids = tokenizer.encode(input_prompt, return_tensors="pt")
    output = model.generate(input_ids, max_length=100)
    generated_text = tokenizer.decode(output[0])
    return generated_text


def main():
    # NOT WORKING
    # st.set_page_config(page_title="Foreign Language Story Generator", layout="centered") # NOT WORKING
    st.title("Foreign Language Story Generator")

    # Load the pre-trained GPT-2 model
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    # Instatniate the GPT-2 tokenizer - responsible for encoding and decoding text 
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # Generate text
    input_prompt = "Once upon a time, there was a young woman named Alice who lived in a small town."


    # Not setup yet
    if st.button("Generate New Story"):
        generated_story = generate_story(input_prompt, model, tokenizer)
        st.text(generated_story)


if __name__ == "__main__":
    main()