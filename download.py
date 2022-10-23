# In this file, we define download_model
# It runs during container build time to get model weights built into the container

# In this example: A Huggingface BERT model

from transformers import T5Tokenizer, T5ForConditionalGeneration

def download_model():
    # do a dry run of loading the huggingface model, which will download weights
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xxl")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xxl", device_map="auto", load_in_8bit=True)
    input_text = "translate English to German: How old are you?"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")


if __name__ == "__main__":
    download_model()