# In this file, we define download_model
# It runs during container build time to get model weights built into the container

# In this example: A Huggingface BERT model
import torch

from transformers import T5Tokenizer, T5ForConditionalGeneration

"""You cannot use CUDA here, because the driver is not available during container build time."""
def download_model():
    # do a dry run of loading the huggingface model, which will download weights
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xxl")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xxl", torch_dtype=torch.float16)


if __name__ == "__main__":
    download_model()