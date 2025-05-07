

import sys, torch
from transformers import (AutoTokenizer,
                          AutoModelForCausalLM)

MODEL_DIR = "./shiritori-model"

tokenizer   = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR,
            torch_dtype=torch.float16,     # fp16 推論
            device_map="auto"
        )



def get_answer(prev_word: str) -> str:
    ids    = tokenizer(prev_word+" → ", return_tensors="pt").to(model.device)
    out    = model.generate(
                **ids,
                max_new_tokens=64,
                do_sample=True,
                top_p=0.9,
                temperature=0.8,
             )
    text = tokenizer.decode(out[0], skip_special_tokens=True)

    return text

if __name__ == "__main__":
    word = sys.argv[1] if len(sys.argv) > 1 else "りんご"
    print(get_answer(word))
