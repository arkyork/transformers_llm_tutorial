

import sys, torch
from transformers import AutoTokenizer,AutoModelForCausalLM,pipeline
from transformers import set_seed


#MODEL_DIR = "./scripts/shiritori-qwen-model"
#MODEL_DIR = "./scripts/shiritori-gemma-model"

MODEL_DIR = "google/gemma-3-1b-it"

model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR,
            device_map="auto",
            attn_implementation="eager"
        )

tokenizer   = AutoTokenizer.from_pretrained(MODEL_DIR)

## pipelineで読み込む
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)



def get_answer(prev_word: str) -> str:
# 推論

    messages = [
        {"role": "system", "content": "あなたは「しりとり」が得意です。「→」で渡された単語でしりとりをして。"},
        {"role": "user",   "content": prev_word+" → "}
    ]
    
    set_seed(144)

    result = generator(prev_word+" → ", 
                        max_new_tokens=24)
    try:
        return result[0]["generated_text"][2]["content"]
    except:
        return result[0]["generated_text"]



if __name__ == "__main__":
    word = sys.argv[1] if len(sys.argv) > 1 else "ぎんこう"
    print(get_answer(word))
