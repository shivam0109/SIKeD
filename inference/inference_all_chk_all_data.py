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
import re
import glob

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)


# Instruction Prompt for individual strategy 
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Solve the following Math Word Problem

### Input:
{}

### Response:
{}"""

# Instruction prompt for Select-sample 
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
    

# Load GSM8K Test 
def load_gsm8k_test():
    ds = datasets.load_dataset('gsm8k','main',split='test')
    df = ds.to_pandas()
    df = df.reset_index(drop=False)
    df.rename(columns={'index':'id'}, inplace=True)
    df['num_answer'] = df['answer'].apply(lambda x: float(x.split("\n")[-1].replace("#",'').replace(",", "").replace(" ", "").replace("\n", "").replace("$", "").replace("x", "").strip()))
    df = df[['id','question','num_answer']].copy()
    df.rename(columns={'num_answer':'answer'}, inplace=True)
    print("GSM8K Test Loaded")
    print("Shape of GSM8K Test: \n", df.shape) 
    return df 


# Load SVAMP Dataset 
def load_svamp(path):
    with open(path, 'r') as f:
        data = json.load(f)
    df = pd.json_normalize(data)
    df['question'] = df['Body'] + " " + df['Question']
    df.rename(columns={'ID':'id', 'Answer':'answer'}, inplace=True)
    print("SVAMP Dataset Loaded")
    print("Shape of SVAMP: \n", df.shape)
    return df[['id','question','answer']].copy() 


def extract_numeric_value(text):
    # Use a regular expression to find the first occurrence of a numeric value in the string
    match = re.search(r'\d+', text)
    if match:
        return float(match.group())  # Convert the matched string to a float
    return None  # Return None if no numeric value is found

# Function to get AsDiv answers 
def extract_asdiv_answer(df_asdiv):
    num_answers = [] 
    for idx, row in df_asdiv.iterrows():
        try:
            answer = float(row['answer'])
            num_answers.append(answer)
        except:
            answer = extract_numeric_value(row['Formula'].split('=')[-1])
            if answer is not None:
                num_answers.append(answer)
            else:
                num_answers.append(-2048)
    print("AsDiV answers not found: ", num_answers.count(-2048))
    return num_answers 

# Load AsDiV 
def load_asdiv(path):
    tree = ET.parse(path)
    root = tree.getroot()
    # Create an empty list to store the data
    data = []

    # Iterate through each Problem in the XML file
    for problem in root.find('ProblemSet').findall('Problem'):
        problem_data = {
            'ID': problem.get('ID'),
            'Grade': problem.get('Grade'),
            'Source': problem.get('Source'),
            'Body': problem.find('Body').text,
            'Question': problem.find('Question').text,
            'Solution-Type': problem.find('Solution-Type').text,
            'Answer': problem.find('Answer').text,
            'Formula': problem.find('Formula').text
        }
        data.append(problem_data)
    
    # Convert to DataFrame 
    df = pd.DataFrame(data)
    df['question'] = df['Body'] + " " + df['Question']
    # Get answer from 'Answer' column 
    df['answer'] = df['Answer'].str.replace(r'\s*\(.*?\)', '', regex=True)
    # if answer is not available from 'Answer', use 'Formula' 
    df['answer'] = extract_asdiv_answer(df)
    df.rename(columns = {'ID':'id'}, inplace=True)
    print("ASDiv loaded")
    print("Shape ASDiv: \n", df.shape)
    return df[['id','question','answer']].copy()


# Load MultiAirth 
def load_multiarith():
    ds = datasets.load_dataset('ChilleD/MultiArith', split='test')
    df = ds.to_pandas()
    df.reset_index(drop=False, inplace=True)
    df.rename(columns={'final_ans':'answer', 'index':'id'}, inplace=True)
    print("MultiArith Loaded")
    print("Shape MultiArith: \n", df.shape)
    return df 

# Load Hendrycks MATH
def load_math(path):
    df = pd.read_csv(path)
    df.reset_index(drop=False, inplace=True)
    df.rename(columns={'index':'id', 'problem':'question', 'solution':'answer'}, inplace=True)
    return df[['id','question','answer']].copy()

# Function to format question for inference 
def format_question(q, prompt):
    return prompt.format(q, "")


def create_batched_questions(df, colname, batch_size=4):
    questions = df[colname].tolist()
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
def save_model_for_vllm(chk_path, max_seq_length, dtype, load_in_4_bit):
    if os.path.exists(os.path.join(chk_path + '_vllm')):
        return 
    if os.path.exists(os.path.join(chk_path + '_merged')):
        return 
    
    # Load model for inference 
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = chk_path, 
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4_bit,
    )
    # Save model for inference with VLLM 
    model.save_pretrained_merged(os.path.join(chk_path + '_vllm'), tokenizer)
    
    # Free Memory 
    torch.cuda.empty_cache()
    del model 
    del tokenizer
    gc.collect()


# Load Model for VLLM Inference 
def load_model_for_vllm(chk_path, max_seq_length):
    if os.path.exists(os.path.join(chk_path + '_vllm')):
        llm = LLM(model=os.path.join(chk_path + '_vllm'))
    else:
        llm = LLM(model=os.path.join(chk_path + '_merged'))
    sampling_params = SamplingParams(temperature=0, max_tokens=max_seq_length)
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


def predict_from_checkpoint_vllm(chk_path, model_dir, batched_questions, df, dataset, config):
    print(chk_path)
    
    save_model_for_vllm(chk_path=chk_path, max_seq_length=config['max_seq_length'], 
                        dtype=config['dtype'], load_in_4_bit=config['load_in_4_bit'])
    
    llm, sampling_params = load_model_for_vllm(chk_path, config['max_seq_length'])
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
    save_path = os.path.join(model_dir, dataset + '_predictions_' + chk_path.split("/")[-1] + ".csv")
    df.to_csv(save_path, index=False)


def check_datasets_processed(model_dir):
    datasets_processed = [] 
    if 'asdiv_predictions_final_model.csv' in os.listdir(model_dir):
        datasets_processed.append('asdiv')
    if 'svamp_predictions_final_model.csv' in os.listdir(model_dir):
        datasets_processed.append('svamp')
    if 'gsm8k_predictions_final_model.csv' in os.listdir(model_dir):
        datasets_processed.append('gsm8k')
    if 'multiarith_predictions_final_model.csv' in os.listdir(model_dir):
        datasets_processed.append('multiarith')
    if 'math_predictions_final_model.csv' in os.listdir(model_dir):
        datasets_processed.append('math')
    return datasets_processed

def check_checkpoints_processed(df_name, model_dir):
    pattern = os.path.join(model_dir, f"{df_name}_predictions_*.csv")
    # Use glob to find all matching files
    matching_files = glob.glob(pattern)
    # Get Processed checkpoints 
    checkpoints_processed = [x.split('/')[-1].split('.')[0].replace('{f}_predictions_'.format(f=df_name),'') for x in matching_files]
    return checkpoints_processed

def run(config_path):
    # Load Config
    config = load_config(config_path)
    model_dir = config['model_dir']

    # Get all datasets 
    dfs = {'asdiv':None, 'svamp':None, 'gsm8k':None, 'multiarith':None, 'math':None}
    dfs['asdiv'] = load_asdiv(config['asdiv_path'])
    dfs['svamp'] = load_svamp(config['svamp_path'])
    dfs['gsm8k'] = load_gsm8k_test()
    dfs['multiarith'] = load_multiarith()
    dfs['math'] = load_math(config['math_path'])
    
    # Check if some datasets have been processed and the current run is a continuation
    datasets_processed = check_datasets_processed(model_dir)

    # Remove keys from dfs if they are present in datasets_processed
    dfs = {key: value for key, value in dfs.items() if key not in datasets_processed}

    for df_name, df in dfs.items():
        # Format Questions for inference 
        if config['strategy'] in ['cot','pot','l2m','ao']:
            df['input'] = df['question'].apply(lambda q: format_question(q, alpaca_prompt))
        else:
            df['input'] = df['question'].apply(lambda q: format_question(q, alpaca_prompt_ss))
        
        # Batch questions 
        batched_questions = create_batched_questions(df, 'input', batch_size=len(df))
        
        # Get all checkpoints 
        checkpoint_paths = [x for x in os.listdir(model_dir) if x.startswith('checkpoint')] + ['final_model']

        # Check if some checkpoints are already processed 
        checkpoints_processed = check_checkpoints_processed(df_name, model_dir)

        # Remaining checkpoints 
        checkpoint_paths = [x for x in checkpoint_paths if x not in checkpoints_processed]

        # Predict and Save using VLLM 
        for checkpoint_path in checkpoint_paths:
            chk_path = os.path.join(model_dir, checkpoint_path)
            predict_from_checkpoint_vllm(chk_path, model_dir, batched_questions, df, df_name, config)
            # Delete the folder created for saving model_for_vllm
            if os.path.exists(os.path.join(chk_path + '_vllm')):
                delete_folder(os.path.join(chk_path + '_vllm'))
            if os.path.exists(os.path.join(chk_path + '_merged')):
                delete_folder(os.path.join(chk_path + '_merged'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unsloth LoRA Inference")
    parser.add_argument("config_path", type=str, help="Path to the configuration file")
    args = parser.parse_args()
    run(args.config_path)