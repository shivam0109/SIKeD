"""
Code for inference on different datasets 
"""

from vllm import LLM, SamplingParams
import pandas as pd 
import transformers 
import torch 
import numpy as np
import pandas as pd 
import datasets 
import random 
import os 
import shutil 
from unsloth import FastLanguageModel
import gc 
from tqdm import tqdm
import yaml 
import json
import xml.etree.ElementTree as ET
from time import time 
import argparse 

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)


# Instruction Prompt for forced strategy inference (selsamp models)
alpaca_prompt_forced = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Solve the following Math Word Problem according to the given strategy.

### Input:
{}

### Response:
{}"""

# Instruction Prompt for general inference (base models)
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Solve the following Math Word Problem

### Input:
{}

### Response:
{}"""

# Instruction prompt for Select-sample - Used while training 
alpaca_prompt_ss = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Solve the following Math Word Problem by firstly genrating the strategy and then solving the problem according to the generated strategy. 

### Input:
{}

### Response:
{}"""


# Load Config File 
def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        return config 
    

# Load GSM8K train 
def load_gsm8k(split):
    ds = datasets.load_dataset('gsm8k','main',split=split)
    df = ds.to_pandas()
    df = df.reset_index(drop=False)
    df.rename(columns={'index':'id'}, inplace=True)
    df['num_answer'] = df['answer'].apply(lambda x: float(x.split("\n")[-1].replace("#",'').replace(",", "").replace(" ", "").replace("\n", "").replace("$", "").replace("x", "").strip()))
    df = df[['id','question','num_answer']].copy()
    df.rename(columns={'num_answer':'answer'}, inplace=True)
    print("GSM8K train Loaded")
    print("Shape of GSM8K train: \n", df.shape) 
    return df 

# Function to format question for inference - forced strategy 
def format_question_forced(q, strategy, prompt):
    return prompt.format(q, "[" + strategy + "]: ")

# Function to format question for inference 
def format_question(q, prompt):
    return prompt.format(q, "")

def create_batched_questions(df, colname, batch_size=4):
    questions = df[colname].tolist()
    if batch_size is None:
        return questions
    batched_questions = [questions[i:i + batch_size] for i in range(0, len(questions), batch_size)]
    return batched_questions


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
    model.save_pretrained_merged('temp_vllm', tokenizer)
    
    # Free Memory 
    torch.cuda.empty_cache()
    del model 
    del tokenizer
    gc.collect()


# Load Model for VLLM Inference 
def load_model_for_vllm(num_samples_to_generate):
    stop_tokens = ["Input", "Input:", "<eot_id>", "<|im_end|>"]
    llm = LLM(model='temp_vllm')
    sampling_params = SamplingParams(temperature=0.7, n=num_samples_to_generate, top_p=0.95, max_tokens=512, stop=stop_tokens)
    # sampling_params = SamplingParams(temperature=0.7, n=10, top_p=0.95, 
    #                                  max_tokens=config['max_seq_length'])
    return llm, sampling_params


# Get predictions using VLLM 
def get_predictions_vllm(llm, sampling_params, all_prompt_formatted):
    outputs = llm.generate(all_prompt_formatted, sampling_params)
    completions = []
    for output in outputs:
        temp = []
        for i in range(len(output.outputs)):
            temp.append(output.outputs[i].text)
        completions.append(temp)
    print (len(completions))
    
    # Free Memory 
    torch.cuda.empty_cache()
    del llm
    del sampling_params 
    gc.collect()
    
    return completions 


def predict_from_checkpoint_vllm(chk_path, batched_questions, df, dataset, config):
    print(chk_path)
    save_model_for_vllm(config)
    llm, sampling_params = load_model_for_vllm(config['num_generation_rounds'])
    print("Getting Predictions....")
    predictions = get_predictions_vllm(llm, sampling_params, batched_questions)
    # Check if predictions have same length as ds 
    try:
        assert(len(predictions)==len(df))
    except:
        print("Prediction and DS Length mismatch")
        print("Length of predictions: ", len(predictions))
        print("Predictions: \n", predictions)
        print("Length of DS: ", len(df))
        print(df.head())
        
    # Add predictions to dataframe 
    df['predictions'] = predictions 
    
    # Save predictions 
    save_path = os.path.join(config['save_predictions_dir'], 
                             'df_forced_preds_' + config['forced_strategy'] + '_' + config['split'] + '.csv')
    df.to_csv(save_path, index=False)
    
    # Delete the folder created for saving model_for_vllm
    delete_folder('temp_vllm')


def run(config_path):
    # Load Config
    config = load_config(config_path)
    
    # Get Data
    dataset = config['data'] # Accepts only GSM8K
    split = config['split']
    if 'gsm8k' in dataset:
        df = load_gsm8k(split)
    if config['test_run']:
        df = df.sample(n=10, random_state=42)

    # Format Questions for inference 
    strat = config['forced_strategy']
    if ('selsamp-cot' in config['checkpoint_path']) or ('selsamp-pot' in config['checkpoint_path']) or ('selsamp-l2m' in config['checkpoint_path']):
        df['input'] = df['question'].apply(lambda q: format_question(q, alpaca_prompt_ss))
    elif 'selsamp' in config['checkpoint_path']:
        # Done for Gemma-7B
        # df['input'] = df['question'].apply(lambda q: format_question_forced(q, strat, alpaca_prompt_forced))
        # Qwen-2
        df['input'] = df['question'].apply(lambda q: format_question_forced(q, strat, alpaca_prompt_ss))
    else:
        df['input'] = df['question'].apply(lambda q: format_question(q, alpaca_prompt))
    
    # Batch questions 
    batched_questions = create_batched_questions(df, 'input', batch_size=None)
    
    # Predict and Save using VLLM 
    chk_path = config['checkpoint_path']
    predict_from_checkpoint_vllm(chk_path, batched_questions, df, dataset, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unsloth LoRA Inference")
    parser.add_argument("config_path", type=str, help="Path to the configuration file")
    args = parser.parse_args()
    run(args.config_path)