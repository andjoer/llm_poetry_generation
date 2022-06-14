from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from parameters import gtp2_model

device = "cuda"
#model_id = "Anjoe/german-poetry-gpt2"
model = GPT2LMHeadModel.from_pretrained(gtp2_model).to(device)
tokenizer = GPT2TokenizerFast.from_pretrained(gtp2_model)


def perplexity(text):
    encodings = tokenizer(text, return_tensors="pt")
    import torch
    from tqdm import tqdm

    max_length = model.config.n_positions
    stride = 512

    nlls = []
    for i in range(0, encodings.input_ids.size(1), stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.input_ids.size(1))
        trg_len = end_loc - i  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs[0] * trg_len

        nlls.append(neg_log_likelihood)

    return torch.exp(torch.stack(nlls).sum() / end_loc).cpu().detach().numpy()
