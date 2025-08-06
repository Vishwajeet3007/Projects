import torch
from transformers import AutoTokenizer
from model.transformer_model import TransformerNextWordPredictor

def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = TransformerNextWordPredictor(vocab_size=tokenizer.vocab_size)
    model.load_state_dict(torch.load("model_weights/transformer_model.pt", map_location="cpu"))
    return model, tokenizer
