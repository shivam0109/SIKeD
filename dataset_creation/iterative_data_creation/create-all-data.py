"""
Code to create all dataset 
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
    print("SLM CoT: ", dfs['cot'].shape)
    print("SLM PoT: ", dfs['pot'].shape)
    print("SLM L2M: ", dfs['l2m'].shape)
    return dfs 


def read_llm_dfs(llm_dir_path):
    dfs = {'cot':None, 'pot':None, 'l2m':None}
    dfs['cot'] = pd.read_csv(os.path.join(llm_dir_path, 'df_cot_kd.csv'))
    dfs['pot'] = pd.read_csv(os.path.join(llm_dir_path, 'df_pot_kd.csv'))
    dfs['l2m'] = pd.read_csv(os.path.join(llm_dir_path, 'df_l2m_kd.csv'))
    print("LLM CoT: ", dfs['cot'].shape)
    print("LLM PoT: ", dfs['pot'].shape)
    print("LLM L2M: ", dfs['l2m'].shape)
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


def get_all_data(df_slm, df_llm):
    # Get input column for LLM 
    df_llm_to_add = df_llm.copy()
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
    df_all = pd.concat([df_slm, df_llm_to_add])
    print("Shape after mixing LLM and SLM: ", df_all.shape)
    
    # Reset index and recreate 'id' column 
    df_all.drop(columns=['id'], inplace=True)
    df_all.reset_index(drop=False, inplace=True)
    df_all.rename(columns={'index':'id'}, inplace=True)

    return df_all 


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
    dfs_all = {'cot':None, 'pot':None, 'l2m':None, 'combined':None, 'cot-biased':None, 'pot-biased':None, 'l2m-biased':None}

    # Get 'All' Data 
    print("CoT")
    dfs_all['cot'] = get_all_data(dfs_slm['cot'], dfs_llm['cot'])
    print("PoT")
    dfs_all['pot'] = get_all_data(dfs_slm['pot'], dfs_llm['pot'])
    print("L2M")
    dfs_all['l2m'] = get_all_data(dfs_slm['l2m'], dfs_llm['l2m'])

    # Concat 'All' Data 
    dfs_all['combined'] = pd.concat([dfs_all['cot'], dfs_all['pot'], dfs_all['l2m']], axis=0).reset_index(drop=True)
    
    # Biased Data 
    dfs_all['cot-biased'] = get_biased_data(dfs_all['combined'], 'cot')
    dfs_all['pot-biased'] = get_biased_data(dfs_all['combined'], 'pot')
    dfs_all['l2m-biased'] = get_biased_data(dfs_all['combined'], 'l2m')

    # Save all data
    for strategy, df in dfs_all.items():
        df.rename(columns={'predictions':'output_answer', 'answer':'correct_answer', 'predicted_num_answer':'llm_numeric_ans'}, inplace=True)
        df.to_csv(os.path.join(save_dir, 'df-all-{strat}-epoch{ep}.csv'.format(strat=strategy, ep=epoch)), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create All Data")
    parser.add_argument("config_path", type=str, help="Path to config file")
    args = parser.parse_args()
    run(args.config_path)