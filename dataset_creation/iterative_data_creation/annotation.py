"""
Code to annotate training dataset using model checkpoints

Step 1: Load GSM8K Train 
Step 2: Load Model Checkpoints (select-sample-biased checkpoints) 
Step 3: Add strategy in format_input functions 
Step 4: Annotate using Select-Sample - 10 generations for each training sample 
"""

import pandas as pd 
import transformers 
import torch 
import numpy as np
import pandas as pd 
import datasets 
import random 
import os 
from unsloth import FastLanguageModel
import gc 
from tqdm import tqdm
from vllm import LLM, SamplingParams
import yaml
import shutil 
from time import time 
import argparse 

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)


# Instruction prompt for Select-sample 
alpaca_prompt_ss = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Solve the following Math Word Problem according to the given strategy. 

### Input:
{}

### Response:
{}"""


# Load Config File 
def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        return config 

# Load GSM8K Training Data 
def load_gsm8k_train(num_rounds):
    # Load GSM8k Train 
    ds = datasets.load_dataset('gsm8k','main',split='train')
    
    # Convert to Pandas DF 
    df = ds.to_pandas()
    df = df.reset_index(drop=False)
    df.rename(columns={'index':'id'}, inplace=True)
    df['num_answer'] = df['answer'].apply(lambda x: float(x.split("\n")[-1].replace("#",'').replace(",", "").replace(" ", "").replace("\n", "").replace("$", "").replace("x", "").strip()))
    df = df[['id','question','num_answer']].copy()
    df.rename(columns={'num_answer':'answer'}, inplace=True)
    
    # Concatenate to get 'num_rounds' dataframe 
    dfs = [df] * num_rounds
    df_concat = pd.concat(dfs, ignore_index=True)
    ds_concat = datasets.Dataset.from_pandas(df_concat)
    print("GSM8K Train Loaded")
    print("Shape of GSM8K Train: \n", len(ds_concat)) 
    print('Unique samples: \n', df_concat.drop_duplicates().shape)
    return ds_concat 

# Get batched inference inputs 
def get_inference_inputs(ds, prompt, strategy, batch_size=32):
    df = ds.to_pandas()
    df['formatted_question'] = df['question'].apply(lambda x: prompt.format(x, '[' + strategy + ']:'))
    questions = df['formatted_question'].tolist()
    if batch_size:
        questions_batched = [questions[i:i + batch_size] for i in range(0, len(questions), batch_size)]
        return questions_batched 
    return questions 

## Helper function for deleting a folder 
def delete_folder(dir_path):
    """
    Deletes a folder and all its contents.
    
    Parameters:
    dir_path (str): The path to the directory to be deleted.
    """
    try:
        shutil.rmtree(dir_path)
        print(f"Directory {dir_path} and all its contents deleted successfully")
    except FileNotFoundError:
        print(f"Error: Directory {dir_path} not found")
    except PermissionError:
        print(f"Error: Permission denied to delete {dir_path}")
    except OSError as e:
        print(f"Error: {e.strerror}")

# Save model for vllm inference 
def save_model_for_vllm(config):
    # Load model for inference 
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = config['checkpoint_path'], 
        max_seq_length = config['max_seq_length'],
        dtype = config['dtype'],
        load_in_4bit = config['load_in_4_bit'],
    )
    # Save model for inference with VLLM 
    model.save_pretrained_merged(os.path.join(config['model_dir'], 'vllm_' + config['strategy']), tokenizer)
    
    # Free Memory 
    torch.cuda.empty_cache()
    del model 
    del tokenizer
    gc.collect()

# Load Model for VLLM Inference 
# Change temperature to 0.7 for multiple generations 
def load_model_for_vllm(chk_path, config):
    llm = LLM(model=os.path.join(config['model_dir'], 'vllm_' + config['strategy'] ))
    sampling_params = SamplingParams(temperature=0.7, max_tokens=config['max_seq_length'])
    return llm, sampling_params

# Get predictions using VLLM 
def get_predictions_vllm(llm, sampling_params, batched_inputs):
    predictions = []
    for batched_input in batched_inputs:
        outputs = llm.generate(batched_input, sampling_params)
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            predictions.append(prompt + "\n" + generated_text)
    
    # Free Memory 
    torch.cuda.empty_cache()
    del llm
    del sampling_params 
    gc.collect()
    
    return predictions 

# Predict using VLLM 
def predict_from_checkpoint_vllm(batched_questions, ds, config):
    chk_path = config['checkpoint_path']
    model_dir = config['model_dir']
    print(chk_path)
    
    save_model_for_vllm(config)
    llm, sampling_params = load_model_for_vllm(chk_path, config)
    
    print("Getting Predictions....")
    predictions = get_predictions_vllm(llm, sampling_params, batched_questions)
    # Check if predictions have same length as ds 
    try:
        assert(len(predictions)==len(ds))
    except:
        print("Prediction and DS Length mismatch")
        print("Length of predictions: ", len(predictions))
        print("Predictions: \n", predictions)
        print("Length of DS: ", len(ds))
        print(ds)
        
    # Add predictions to dataframe
    ds = ds.add_column("predictions", predictions)
    
    # Save predictions 
    save_path = os.path.join(model_dir, 'gsm8k_annotation_' + config['strategy'])
    ds.save_to_disk(save_path)
    
    # Delete the folder created for saving model_for_vllm
    delete_folder(os.path.join(config['model_dir'], 'vllm_' + config['strategy']))


def run(config_path):
    # Load Config 
    config = load_config(config_path)
    
    # Get GSM8K Training Data 
    ds = load_gsm8k_train(num_rounds=config['num_generation_rounds'])
    print("Data loaded")
    print(ds)
    
    # Get batched inputs 
    batched_questions = get_inference_inputs(ds=ds, prompt=alpaca_prompt_ss, 
                                             strategy=config['strategy'], batch_size=4)
    
    # Get Predictions  
    predict_from_checkpoint_vllm(batched_questions, ds, config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vllm Val Inference")
    parser.add_argument("config_path", type=str, help="Path to the configuration file")
    args = parser.parse_args()
    run(args.config_path)
