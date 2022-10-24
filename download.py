import torch

from transformers import T5Tokenizer, T5ForConditionalGeneration


def download_model():
    """You cannot use CUDA here, because the driver is not available during container build time."""
    # do a dry run of loading the huggingface model, which will download weights
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xxl")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xxl", device_map="auto", load_in_8bit=True)


if __name__ == "__main__":
    download_model()
