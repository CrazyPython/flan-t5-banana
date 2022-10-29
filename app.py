from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch


# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model, tokenizer

    device = 0 if torch.cuda.is_available() else -1
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xxl")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl", device_map="auto", torch_dtype=torch.bfloat16)

    input_text = "translate English to German: How old are you?"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")


# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs: dict) -> dict:
    global model

    # Parse out your arguments
    prompt = model_inputs.get('prompt', None)
    if prompt == None:
        return {'message': "No prompt provided"}

    # Run the model
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to('cuda')
    outputs = model.generate(input_ids)
    result = tokenizer.decode(outputs[0])

    # Return the results as a dictionary
    return {
        "text": result
    }
