"""
Create Dataframe from LLama Annotations 

The output dataframe will have
1. Reasoning chains expanded as individual rows. 
2. No duplicate (exactly duplicate) reasoning chains. 
3. Columns: Question, Strategy + Description, Reasoning Chain, Answer (majority by LLama), Answer (true)

Usage: python create_data.py path/to/config_file.yaml
"""

import yaml 
import pandas as pd 
import json 
import argparse
from typing import List, Dict 
import numpy as np 
import re 
from math_solver import solve_mwp
import signal 
SEED = 42 


# Load Config File 
def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        return config 


# Helper function to read data 
def read_data_helper(path:str) -> List[Dict[str,any]]:
    try:
        with open(path, 'r') as json_file:
            data = [json.loads(line) for line in json_file]
    except:
        with open(path, 'r') as f:
            data_string = f.read()
        data = [] 
        lines = data_string.splitlines()
        for i,line in enumerate(lines):
            try:
                json_data = json.loads(line)
                data.append(json_data)
            except json.JSONDecodeError as e:
                print("Error decoding JSON at line {l}".format(l=i+1))

    return data 


# Create Dataframe from Json-List 
def df_from_json(data:List[Dict[str, any]], strat:str=None, fields:List=["question","answer","sample"], expand_samples=False) -> pd.DataFrame:
    if expand_samples:
        rows = [] 
        for idx, entry in enumerate(data):
            question = entry[fields[0]]
            answer = entry[fields[1]]
            samples = entry[fields[2]]
            for sample in samples:
                rows.append({
                    'id': idx, 
                    'question': question,
                    'answer': answer,
                    'sample': sample,
                    'strategy': strat
                })
        return pd.DataFrame(rows)

    else:
        # Empty dictionary of fields 
        field_dict = dict()
        for f in fields:
            field_dict[f] = []
        
        # Append data to each f 
        for i in range(len(data)):
            for field in fields:
                # Take the first sample from LLM generations (sample)
                if field=="sample":
                    field_dict[field].append(data[i][field][0])
                else:
                    field_dict[field].append(data[i][field])
        
        # Add strategy column
        if strat is not None:
            field_dict["strategy"] = [strat] * len(data)
        # Add 'ID' column 
        if 'id' not in fields:
            field_dict["id"] = [x for x in range(len(data))]
        return pd.DataFrame(field_dict)


# Function to read data 
def read_data(path:str, strat:str, expand_samples:bool, fields:List) -> pd.DataFrame:
    data = read_data_helper(path)
    df = df_from_json(data=data, strat=strat, expand_samples=expand_samples, fields=fields)
    return df 


# Function to extract predictions - 'answers' field 
def extract_pred(pred_sample):
    all_pred = []
    for sample in pred_sample:
        # val = sample.split("The answer is ")[-1]
        pattern = r'[$]?[-+]?\d+(?:\.\d+)?(?:,\d+)*[$]?'
        matches = re.findall(pattern, sample)
        if  matches != []:
            all_pred.append(float(matches[-1].replace(",", "").replace(" ", "").replace("\n", "").replace("$", "").replace("x", "")))
        else:
            all_pred.append(-2048.0)
            print ("No answer found!")
            print (sample)
    return all_pred


# Timeout Handler 
def timeout_handler(signum, frame):
    raise TimeoutError("Execution timed out")


# Function to extract predictions using Math Solver - for noNL and NL+SL 
def extract_pred_math_solver(pred_sample):
    time_limit = 10 
    ans = [] 
    for pred in pred_sample:
        signal.signal(signal.SIGALRM, timeout_handler)
        try:
            signal.alarm(time_limit)
            ans.append(solve_mwp(pred))
            signal.alarm(0)
        except TimeoutError:
            print("Function execution timed out")
            ans.append("[invalid]")
        finally:
            signal.alarm(0)
    assert(len(ans)==len(pred_sample))
    return ans 


# Function to drop rows that have not been annotated 
def drop_rows(df:pd.DataFrame):
    # No answer found 
    df = df[df['llm_numeric_ans']!=-2048].copy()
    print("Shape after removing rows where no answers were found: ", df.shape)

    # '[invalid]' answers
    prev_shape = df.shape[0]
    df = df[df['llm_numeric_ans']!='[invalid]'].copy()
    new_shape = df.shape[0]
    print("Dropping {nrows} rows where answer is '[invalid]'".format(nrows = prev_shape-new_shape))
    print("Shape after dropping '[invalid]' rows: ", df.shape)

    df.reset_index(drop=True, inplace=True)
    return df 


# Function to extract numeric answer from LLM responses and add 'is_correct' column 
def get_llm_answer(df:pd.DataFrame, strat:str, correct_answer_col:str='answer', llm_response_col:str='sample'):
    print("Strategy: ", strat)
    print("DF Shape: ", df.shape)

    # Get llm numeric answer 
    if strat in ['cot','l2m']:
        df['llm_numeric_ans'] = extract_pred(df[llm_response_col])
    else:
        df['llm_numeric_ans'] = extract_pred_math_solver(df[llm_response_col])
    
    ans_not_found = (df['llm_numeric_ans']==-2048).sum()
    perc_ans_not_found = np.round(100*ans_not_found/len(df),2)
    print("{n} samples with no answer. {x} perc".format(n=ans_not_found, x=perc_ans_not_found))

    # Drop not annotated rows
    df = drop_rows(df)
    
    # Get ground truth answer 
    df['ground_truth'] = df[correct_answer_col].apply(lambda x: x.split('\n')[-1].replace(",", "").replace(" ", "").replace("\n", "").replace("$", "").replace("x", "").replace('#','').strip())
    df['ground_truth'] = df['ground_truth'].apply(float)

    # Add is_correct column 
    df['is_correct'] = df.apply(lambda row: row['llm_numeric_ans']==row['ground_truth'], axis=1)
    print("Shape after adding 'is_correct' column: ", df.shape)
    print("Incidence Rate: ", np.round(100*df['is_correct'].sum()/df.shape[0],2))
    print(df.head())
    return df 


# Function for knowledge distillation - Take only 1s 
def get_kd_data(df:pd.DataFrame) -> pd.DataFrame:
    df_kd = df[df["is_correct"]==True].copy()
    df_kd.drop_duplicates(subset=['id'], inplace=True)
    df_kd.reset_index(drop=True, inplace=True)
    df_kd = df_kd.sample(frac=1, random_state=SEED)
    print("Shape: Distilled Dataset", df_kd.shape)
    print("Columns: \n", df_kd.columns) 
    print(df_kd.head())
    return df_kd 


# Function to get a biased dataframe according to priority dictionary 
def get_biased_kd_data(df, priority_dict):
    # Add a new column to DataFrame to hold priority values
    df['priority'] = df['strategy'].map(priority_dict)
    # Sort DataFrame by priority
    df = df.sort_values(by='priority')
    # Drop duplicates while keeping the first occurrence
    df = df.drop_duplicates(subset='question', keep='first')
    # Drop the temporary priority column
    df = df.drop(columns='priority')
    return df 


# Function to run for LLaMA-3-70B Output 
def run_llama3(config_path):
    # Load config 
    config = load_config(config_path)
    expand_samples = config['samples']['expand']
    question_col = config['samples']['question_col']
    ground_truth_col = config['samples']['ground_truth_col']
    llm_response_col = config['samples']['llm_response_col']

    # Create a list to store all DFs 
    all_dfs = [] 
    
    # Build DFs from different strategies
    # CoT  
    if config['input']["cot"]:
        df = read_data(path=config['input']['cot'], strat='cot', expand_samples=expand_samples, fields=[question_col, ground_truth_col, llm_response_col])
        df_cot = get_llm_answer(df, strat='cot', correct_answer_col='answer', llm_response_col='sample')
        df_cot.to_csv(config['output']['cot'], index=False)
        all_dfs.append(df_cot)
    
    # PoT 
    if config['input']['pot']:
        df = read_data(path=config['input']['pot'], strat='pot', expand_samples=expand_samples, fields=[question_col, ground_truth_col, llm_response_col])
        df_pot = get_llm_answer(df, strat='pot', correct_answer_col='answer', llm_response_col='sample')
        df_pot.to_csv(config['output']['pot'], index=False)
        all_dfs.append(df_pot)
    
    # L2M 
    if config['input']['l2m']:
        df = read_data(path=config['input']['l2m'], strat='l2m', expand_samples=expand_samples, fields=[question_col, ground_truth_col, llm_response_col])
        df_l2m = get_llm_answer(df, strat='l2m', correct_answer_col='answer', llm_response_col='sample')
        df_l2m.to_csv(config['output']['l2m'], index=False)
        all_dfs.append(df_l2m)
    
    # Concat 
    if config['output']['concat']:
        df_concat = pd.concat(all_dfs)
        df_concat.to_csv(config['output']['concat'], index=False)

    # Knowledge Distillation 
    if config['distill_output']['perform']:
        all_kd_dfs = [] 
        
        # CoT 
        if config['distill_output']['cot']:
            df_kd_cot = get_kd_data(df_cot)
            print("DF KD COT Shape: ", df_kd_cot.shape)
            assert(df_kd_cot.shape[0]==df_kd_cot['is_correct'].sum())
            df_kd_cot.to_csv(config['distill_output']['cot'], index=False)
            all_kd_dfs.append(df_kd_cot)

        # PoT
        if config['distill_output']['pot']:
            df_kd_pot = get_kd_data(df_pot)
            print("DF KD POT Shape: ", df_kd_pot.shape)
            assert(df_kd_pot.shape[0]==df_kd_pot['is_correct'].sum())
            df_kd_pot.to_csv(config['distill_output']['pot'], index=False)
            all_kd_dfs.append(df_kd_pot)

        # L2M 
        if config['distill_output']['l2m']:
            df_kd_l2m = get_kd_data(df_l2m)
            print("DF KD LtM Shape: ", df_kd_l2m.shape)
            assert(df_kd_l2m.shape[0]==df_kd_l2m['is_correct'].sum())
            df_kd_l2m.to_csv(config['distill_output']['l2m'], index=False)
            all_kd_dfs.append(df_kd_l2m)

        # Concat         
        if config['distill_output']['concat']:
            df_kd_concat = pd.concat(all_kd_dfs)
            print("DF KD Concat Shape: ", df_kd_concat.shape)
            print(df_kd_concat.head())
            df_kd_concat.to_csv(config['distill_output']['concat'], index=False)

        # CoT Biased 
        if config['distill_output']['concat_cot_biased']:
            df_kd_concat = pd.concat(all_kd_dfs)
            priority_dict = {'cot':0, 'pot':1, 'l2m':2}
            df_kd_cot_biased = get_biased_kd_data(df_kd_concat, priority_dict)
            print("Shape of CoT Biased KD Data: ", df_kd_cot_biased.shape)
            print("Distribution of Strategies: \n", df_kd_cot_biased[['question','strategy']].groupby('strategy').count())
            df_kd_cot_biased.to_csv(config['distill_output']['concat_cot_biased'], index=False)

        # PoT Biased 
        if config['distill_output']['concat_pot_biased']:
            df_kd_concat = pd.concat(all_kd_dfs)
            priority_dict = {'pot':0, 'cot':1, 'l2m':2}
            df_kd_pot_biased = get_biased_kd_data(df_kd_concat, priority_dict)
            print("Shape of PoT Biased KD Data: ", df_kd_pot_biased.shape)
            print("Distribution of Strategies: \n", df_kd_pot_biased[['question','strategy']].groupby('strategy').count())
            df_kd_pot_biased.to_csv(config['distill_output']['concat_pot_biased'], index=False)

        # L2M Biased 
        if config['distill_output']['concat_l2m_biased']:
            df_kd_concat = pd.concat(all_kd_dfs)
            priority_dict = {'l2m':0, 'pot':1, 'cot':2}
            df_kd_l2m_biased = get_biased_kd_data(df_kd_concat, priority_dict)
            print("Shape of L2M Biased KD Data: ", df_kd_l2m_biased.shape)
            print("Distribution of Strategies: \n", df_kd_l2m_biased[['question','strategy']].groupby('strategy').count())
            df_kd_l2m_biased.to_csv(config['distill_output']['concat_l2m_biased'], index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create Data using the provided config file")
    parser.add_argument("config_file", type=str, help="Path to the configuration file")
    args = parser.parse_args()
    run_llama3(args.config_file)