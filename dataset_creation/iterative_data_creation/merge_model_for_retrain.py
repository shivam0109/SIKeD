import torch 
import numpy as np
import random 
import os 
from unsloth import FastLanguageModel
import gc 
import argparse 

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)


# Save model for vllm inference 
def save_merged_model(chk_path):
    # Load model for inference 
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = chk_path, 
        max_seq_length = 1024,
        dtype = None,
        load_in_4bit = False,
    )
    # Save model for inference with VLLM 
    model.save_pretrained_merged(os.path.join(chk_path + '_merged'), tokenizer)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge LoRA Adapters")
    parser.add_argument("chk_path", type=str, help="Path to the checkpoint")
    args = parser.parse_args()
    save_merged_model(args.chk_path)

