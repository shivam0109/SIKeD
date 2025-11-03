"""
Code to create mixed dataset 
Take generations from SLM and add generations from LLM 
"""

import pandas as pd 
import numpy as np 
import argparse
import os 
import yaml 

# Load Config File 
def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        return config 

def read_slm_dfs(slm_dir_path, epoch):
    dfs = {'cot':None, 'pot':None, 'l2m':None}
    dfs['cot'] = pd.read_csv(os.path.join(slm_dir_path, 'df-cot-distilled-epoch{x}.csv'.format(x=epoch)))
    dfs['pot'] = pd.read_csv(os.path.join(slm_dir_path, 'df-pot-distilled-epoch{x}.csv'.format(x=epoch)))
    dfs['l2m'] = pd.read_csv(os.path.join(slm_dir_path, 'df-l2m-distilled-epoch{x}.csv'.format(x=epoch)))
    return dfs 


def read_llm_dfs(llm_dir_path):
    dfs = {'cot':None, 'pot':None, 'l2m':None}
    dfs['cot'] = pd.read_csv(os.path.join(llm_dir_path, 'df_cot_kd.csv'))
    dfs['pot'] = pd.read_csv(os.path.join(llm_dir_path, 'df_pot_kd.csv'))
    dfs['l2m'] = pd.read_csv(os.path.join(llm_dir_path, 'df_l2m_kd.csv'))
    return dfs 

def get_llm_input_text(question, slm_input_text):
    instruction = slm_input_text.split("\n")[3]
    llm_input_text = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    {}

    ### Input:
    {}

    ### Response:
    {}""".format(instruction, question, "")
    return llm_input_text


def get_mixed_data(df_slm, df_llm):

    # Get Questions not present in SLM 
    questions_slm = set(df_slm['question'])
    questions_llm = set(df_llm['question'])
    questions_to_add = list(questions_llm - questions_slm) 
    questions_slm_only = list(questions_slm - questions_llm) 
    print("Questions solvable through SLM but not LLM: ", len(questions_slm_only))
    print("Questions solvable through LLM but not SLM: ", len(questions_to_add))

    # Get input column for LLM 
    df_llm_to_add = df_llm[df_llm['question'].isin(questions_to_add)]
    slm_input_text = df_slm['input'].iloc[0]
    df_llm_to_add['input'] = df_llm_to_add['question'].apply(lambda x: get_llm_input_text(x, slm_input_text))
    
    # Get correct numeric answer 
    df_llm_to_add['answer'] = df_llm_to_add['answer'].apply(lambda x: float(x.split("\n")[-1].replace("#",'').replace(",", "").replace(" ", "").replace("\n", "").replace("$", "").replace("x", "").strip()))

    # Rename LLM columns to match SLM columns 
    df_llm_to_add.rename(columns={'sample':'predictions', 'llm_numeric_ans':'predicted_num_answer'}, inplace=True)
    llms_col_to_take = ['id', 'question', 'answer', 'input', 'predictions', 'strategy', 'predicted_num_answer', 'is_correct']

    # DF LLM to add 
    df_llm_to_add = df_llm_to_add[llms_col_to_take].copy()

    # Concat 
    df_mixed = pd.concat([df_slm, df_llm_to_add])
    print("Shape after mixing LLM and SLM: ", df_mixed.shape)
    
    # Reset index and recreate 'id' column 
    df_mixed.drop(columns=['id'], inplace=True)
    df_mixed.reset_index(drop=False, inplace=True)
    df_mixed.rename(columns={'index':'id'}, inplace=True)

    return df_mixed 


def get_biased_data(df_concat, strategy):
    df = df_concat.copy()
    if strategy == 'pot':
        sort_map = {'pot':0, 'cot':1, 'l2m':2}
    elif strategy == 'cot':
        sort_map = {'cot':0, 'pot':1, 'l2m':2}
    else:
        sort_map = {'pot':1, 'cot':2, 'l2m':0}
    df['sort_strat'] = df['strategy'].map(sort_map)
    df = df.sort_values(by=['sort_strat'])
    df = df.drop_duplicates(subset=['question'])
    assert(df['is_correct'].sum() == df.shape[0])
    print("Shape: ", df.shape)
    print("Distribution: ")
    print(df.groupby(by=['strategy']).count())
    print("Distribution in percentage: ")
    print(np.round(100 * df.groupby(by=['strategy']).count())/df.shape[0], 2)
    return df.drop(columns=['sort_strat']) 


def run(config_path):
    config = load_config(config_path)
    slm_dir_path = config['SLM_DIR_PATH']
    llm_dir_path = config['LLM_DIR_PATH']
    epoch = config['EPOCH']
    save_dir = config['SAVE_DIR']
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Read Data 
    dfs_slm = read_slm_dfs(slm_dir_path, epoch)
    dfs_llm = read_llm_dfs(llm_dir_path)
    dfs_mixed = {'cot':None, 'pot':None, 'l2m':None, 'combined':None, 'cot-biased':None, 'pot-biased':None, 'l2m-biased':None}

    # Get Mixed Data 
    dfs_mixed['cot'] = get_mixed_data(dfs_slm['cot'], dfs_llm['cot'])
    dfs_mixed['pot'] = get_mixed_data(dfs_slm['pot'], dfs_llm['pot'])
    dfs_mixed['l2m'] = get_mixed_data(dfs_slm['l2m'], dfs_llm['l2m'])

    # Concat Mixed Data 
    dfs_mixed['combined'] = pd.concat([dfs_mixed['cot'], dfs_mixed['pot'], dfs_mixed['l2m']], axis=0).reset_index(drop=True)
    
    # Biased Data 
    dfs_mixed['cot-biased'] = get_biased_data(dfs_mixed['combined'], 'cot')
    dfs_mixed['pot-biased'] = get_biased_data(dfs_mixed['combined'], 'pot')
    dfs_mixed['l2m-biased'] = get_biased_data(dfs_mixed['combined'], 'l2m')

    # Save all data
    for strategy, df in dfs_mixed.items():
        df.rename(columns={'predictions':'output_answer', 'answer':'correct_answer', 'predicted_num_answer':'llm_numeric_ans'}, inplace=True)
        df.to_csv(os.path.join(save_dir, 'df-mixed-{strat}-epoch{ep}.csv'.format(strat=strategy, ep=epoch)), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create Mixed Data")
    parser.add_argument("config_path", type=str, help="Path to config file")
    args = parser.parse_args()
    run(args.config_path)