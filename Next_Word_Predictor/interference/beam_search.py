# next-word-predictor/inference/beam_search.py
import torch
import torch.nn.functional as F
from model.transformer_model import TransformerNextWordPredictor
from transformers import AutoTokenizer

@torch.no_grad()
def beam_search_predict(
    model,
    tokenizer,
    input_text,
    beam_width=3,
    max_len=20,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    model.eval()
    model.to(device)

    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    input_ids = input_ids[:, -10:]  # restrict to last 10 tokens if long

    sequences = [(input_ids, 0)]  # (sequence_tensor, score)

    for _ in range(max_len):
        all_candidates = []
        for seq, score in sequences:
            output = model(seq)
            logits = output[:, -1, :]
            probs = F.log_softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, beam_width, dim=-1)

            for i in range(beam_width):
                next_token = topk_indices[0][i].unsqueeze(0).unsqueeze(0)
                new_seq = torch.cat([seq, next_token], dim=1)
                new_score = score + topk_probs[0][i].item()
                all_candidates.append((new_seq, new_score))

        sequences = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_width]

    final_outputs = []
    for seq, score in sequences:
        text = tokenizer.decode(seq.squeeze(), skip_special_tokens=True)
        final_outputs.append((text, score))

    return final_outputs

# Example usage (after training)
if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = TransformerNextWordPredictor(vocab_size=tokenizer.vocab_size)
    model.load_state_dict(torch.load("model/transformer_model.pt", map_location="cpu"))

    input_prompt = "Once upon a time"
    results = beam_search_predict(model, tokenizer, input_prompt, beam_width=3)
    for text, score in results:
        print(f"Generated: {text}\nScore: {score:.2f}\n")
