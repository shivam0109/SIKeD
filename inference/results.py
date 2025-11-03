import pandas as pd 
import numpy as np
import pandas as pd 
import datasets 
import random 
import torch
import os 
import yaml 
import signal
import argparse 
import re

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# Load Config File 
def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        return config 

## Function to filter for only multiarithematic test set 
def filter_multiarith_test(df_ma):
    if 'split' in list(df_ma.columns):
        df_ma_test = df_ma[df_ma['split']=='test'].copy()
        return df_ma_test 
    return df_ma 


def extract_first_float(string):
    """
    Extracts the first float number from a given string.

    Parameters:
    string (str): The input string.

    Returns:
    float: The first float number found in the string. Returns None if no float number is found.
    """
    # Regex pattern to match floats
    float_pattern = r'[-+]?\d*\.\d+([eE][-+]?\d+)?'
    
    # Search for the first match in the string
    match = re.search(float_pattern, string)
    
    # If a match is found, return it as a float, otherwise return None
    if match:
        return float(match.group())
    else:
        return -2048


## Function to get Predicted Answers in Numeric Format
def get_pred_answers(pred):
    pred = pred.lower().replace("<pad>","").replace("<eos>","").replace("<|im_end|>","")
    pred_lst = pred.split("\n")
    answer = ''
    # Check if 'final answer: ' is present in a string 
    index_final_answer = pred.find('final answer:')
    if index_final_answer != -1:
        content_after_final_answer = pred[index_final_answer + len('final answer:'):]
        answer = content_after_final_answer.replace(",", "").replace(" ", "").replace("\n", "").replace("$", "").replace("x", "").replace('%','').strip()
        try:
            answer = float(answer)
        except:
            print("Cant convert predicted answer to float: {x}".format(x=answer))
            for sentence in pred_lst:
                if sentence.startswith('the answer is') or sentence.startswith("final answer:"):
                    answer = sentence.replace('the answer is','').replace("final answer:","")
                    answer = answer.replace(",", "").replace(" ", "").replace("\n", "").replace("$", "").replace("x", "").replace('%','').strip()
                    if len(answer) > 0 and answer[-1] == '.':
                        answer = answer[:-1]
                    if '=' in answer:
                        answer = answer.split('=')[-1]
                    if '/' in answer:
                        frac = answer.split('/')
                        try:
                            num, den = float(frac[0]),float(frac[1])
                            answer = num/den
                        except:
                            continue        
                    break
    # if answer is still not found, take the last string after '='
    if answer == '':
        answer = pred.lower().split("=")[-1]
    try:
        answer = float(answer)
    except:
        answer = -2048
    return answer 


# Function to get accuracy of predictions - for CoT and L2M
def get_accuracy(df_path):
    df = pd.read_csv(df_path)
    N = df.shape[0]
    if 'multiarith' in df_path:
        df = filter_multiarith_test(df) 
    try:
        df['answer'] = df['answer'].apply(float)
    except:
        df['answer'] = pd.to_numeric(df['answer'], errors='coerce')
        print("Removed {x} rows with non-float true answers".format(x=df['answer'].isnull().sum()))
        df = df.dropna(subset=['answer'])
        
    df['predicted_num_answer'] = df['predictions'].apply(lambda x: get_pred_answers(x))
    not_found = (df['predicted_num_answer']==-2048).sum()
    print("Predicted answer not found for {x} samples".format(x=not_found))
    df['is_correct'] = df.apply(lambda row: float(row['answer'])==float(row['predicted_num_answer']), axis=1)
    acc = (df['is_correct'].sum(), np.round(100*df['is_correct'].sum()/N, 2))
    return acc 


# Math Solver for PoT 
def solve_mwp(completion, prefix_frn='prefix.txt'):
    with open(prefix_frn, 'r') as fr:
        prefix = fr.read()
    completion = completion.rstrip("#")
    code = f"{prefix}\n{completion}"
    try:
        locs = {}
        exec(code, locs, locs)
        answer = locs["answer"]
    except Exception as e:
        print(e)
        answer = "[invalid]"
    return answer 


# Timeout Handler 
def timeout_handler(signum, frame):
    raise TimeoutError("Execution timed out")

# Function to extract predictions using Math Solver - for noNL and NL+SL 
def extract_pred_math_solver(pred):
    time_limit = 10 
    ans = -2048
    signal.signal(signal.SIGALRM, timeout_handler)
    try:
        signal.alarm(time_limit)
        ans = solve_mwp(pred)
        signal.alarm(0)
    except TimeoutError:
        print("Function execution timed out")
        ans = -2048
    finally:
        signal.alarm(0)
    if isinstance(ans, tuple) or isinstance(ans, list):
        ans = ans[0]
    try:
        ans = float(ans)
    except:
        ans = -2048
    return float(ans)

# Helper function to add numeric answer to predictions 
def format_predictions(predictions):
    completions = [x.replace('<pad>','').replace('<eos>','').replace('<|im_end|>','').replace('\n[pot]: ', '').split('Response:')[-1] for x in predictions]
    return completions

def is_correct_helper(true_answer, predicted_answer):
    if isinstance(predicted_answer, list) or isinstance(predicted_answer, tuple):
        predicted_answer = predicted_answer[0]
    if not predicted_answer:
        return False
    try:
        predicted_answer = float(predicted_answer)
    except:
        print("cant typecase {x} into float".format(x=predicted_answer))
        return False
    return float(true_answer) == predicted_answer


def get_accuracy_pot(df_path):
    df = pd.read_csv(df_path)
    N = df.shape[0]
    if 'multiarith' in df_path:
        df = filter_multiarith_test(df) 
    try:
        df['answer'] = df['answer'].apply(float)
    except:
        df['answer'] = pd.to_numeric(df['answer'], errors='coerce')
        print("Removed {x} rows with non-float true answers".format(x=df['answer'].isnull().sum()))
        df = df.dropna(subset=['answer'])
    
    predictions = df['predictions'].tolist()
    # Format predictions 
    formatted_predictions = format_predictions(predictions)
    # Get Numeric answers from formatted predictions 
    predicted_answers = [extract_pred_math_solver(formatted_prediction) for formatted_prediction in formatted_predictions]
    df['predicted_num_answer'] = predicted_answers
    not_found = predicted_answers.count(-2048)
    print("Predicted answer not found for {x} samples".format(x=not_found))
    df['is_correct'] = df.apply(lambda row: is_correct_helper(row['answer'], row['predicted_num_answer']), axis=1)
    acc = (df['is_correct'].sum(), np.round(100*df['is_correct'].sum()/N, 2))
    return acc 


# Function to add predictions to Dataset 
def add_preds_to_df(df):
    num_preds = []
    strategies = [] 
    for idx, row in df.iterrows():
        pred = row['predictions']
        completion = pred.replace('<pad>','').replace('<eos>','').replace("<|im_end|>","").split('Response:')[-1]
        strategy = completion.split(':')[0].replace('\n','').replace('[','').replace(']','')
        reasoning_chain = completion[completion.find(':')+1:].strip()
        num_pred = -2048
        if 'l2m' in strategy or 'cot' in strategy:
            num_pred = get_pred_answers(reasoning_chain)
        elif 'pot' in strategy:
            num_pred = extract_pred_math_solver(reasoning_chain)
        elif 'standard' in strategy:
            num_pred = extract_first_float(reasoning_chain)
        else:
            print("No Strategy found for {i}th sample. Appending -2048".format(i=idx))
        num_preds.append(num_pred)
        strategies.append(strategy)
        
    df['predicted_num_answer'] = num_preds 
    df['predicted_strategy'] = strategies
    print("No predicted answers for {x} samples".format(x=num_preds.count(-2048)))
    print(df.head())
    return df 


def get_strategy_acc(df, strategy):
    df_subset = df[df['predicted_strategy'].apply(lambda x: strategy in x.lower())]
    correct = (df_subset['answer'] == df_subset['predicted_num_answer']).sum()
    total = df_subset.shape[0]
    if total==0:
        acc = 0
    else:
        acc = np.round(100 * correct/total, 2)
    return acc 


def get_accuracy_ss(df_path):
    strat_dict = {'cot_samples':-1, 'pot_samples':-1, 'l2m_samples':-1, 'ao_samples':-1, 
                 'cot_acc':-1, 'pot_acc':-1, 'l2m_acc':-1, 'ao_acc':-1}
    
    df = pd.read_csv(df_path)
    N = df.shape[0]
    if 'multiarith' in df_path:
        df = filter_multiarith_test(df) 
    try:
        df['answer'] = df['answer'].apply(float)
    except:
        df['answer'] = pd.to_numeric(df['answer'], errors='coerce')
        print("Removed {x} rows with non-float true answers".format(x=df['answer'].isnull().sum()))
        df = df.dropna(subset=['answer'])
    
    df = add_preds_to_df(df)
    df['is_correct'] = df.apply(lambda row: float(row['answer'])==float(row['predicted_num_answer']), axis=1)
    acc = (df['is_correct'].sum(), np.round(100*df['is_correct'].sum()/N, 2))
    strat_dict['cot_samples'] = df['predicted_strategy'].apply(lambda x: 'cot' in x.lower()).sum()
    strat_dict['pot_samples'] = df['predicted_strategy'].apply(lambda x: 'pot' in x.lower()).sum()
    strat_dict['l2m_samples'] = df['predicted_strategy'].apply(lambda x: 'l2m' in x.lower()).sum()
    strat_dict['ao_samples'] = df['predicted_strategy'].apply(lambda x: 'ao' in x.lower()).sum()
    strat_dict['cot_acc'] = get_strategy_acc(df, 'cot') 
    strat_dict['pot_acc'] = get_strategy_acc(df, 'pot') 
    strat_dict['l2m_acc'] = get_strategy_acc(df, 'l2m') 
    strat_dict['ao_acc'] = get_strategy_acc(df, 'standard')
    return acc, strat_dict


def run(config_path):
    # Load Config
    config = load_config(config_path)
    num_strats = 0 
    only_one_strat = None 
    # CoT 
    cot_acc = {'GSM8K':-1, 'SVAMP':-1, 'ASDiv':-1, 'MultiArith':-1}
    print("CoT Predictions\n")
    if config['COT']['GSM8K']:
        num_strats += 1
        only_one_strat = 'COT'
        print("GSM8K\n")
        cot_acc['GSM8K'] = get_accuracy(config['COT']['GSM8K'])
    if config['COT']['SVAMP']:
        print("SVAMP\n")
        cot_acc['SVAMP'] = get_accuracy(config['COT']['SVAMP'])
    if config['COT']['ASDiv']:
        print("ASDiv\n")
        cot_acc['ASDiv'] = get_accuracy(config['COT']['ASDiv'])
    if config['COT']['MultiArith']:
        print('MultiArith\n')
        cot_acc['MultiArith'] = get_accuracy(config['COT']['MultiArith'])
    
    # L2M 
    l2m_acc = {'GSM8K':-1, 'SVAMP':-1, 'ASDiv':-1, 'MultiArith':-1}
    print("\nL2M Predictions\n")
    if config['L2M']['GSM8K']:
        num_strats += 1
        only_one_strat = 'L2M'
        print("GSM8K\n")
        l2m_acc['GSM8K'] = get_accuracy(config['L2M']['GSM8K'])
    if config['L2M']['SVAMP']:
        print("SVAMP\n")
        l2m_acc['SVAMP'] = get_accuracy(config['L2M']['SVAMP'])
    if config['L2M']['ASDiv']:
        print("ASDiv\n")
        l2m_acc['ASDiv'] = get_accuracy(config['L2M']['ASDiv'])
    if config['L2M']['MultiArith']:
        print('MultiArith\n')
        l2m_acc['MultiArith'] = get_accuracy(config['L2M']['MultiArith'])
        
    # PoT 
    pot_acc = {'GSM8K':-1, 'SVAMP':-1, 'ASDiv':-1, 'MultiArith':-1}
    print("\nPoT Predictions\n")
    if config['POT']['GSM8K']:
        num_strats += 1
        only_one_strat = 'POT'
        print("GSM8K\n")
        pot_acc['GSM8K'] = get_accuracy_pot(config['POT']['GSM8K'])
    if config['POT']['SVAMP']:
        print("SVAMP\n")
        pot_acc['SVAMP'] = get_accuracy_pot(config['POT']['SVAMP'])
    if config['POT']['ASDiv']:
        print("ASDiv\n")
        pot_acc['ASDiv'] = get_accuracy_pot(config['POT']['ASDiv'])
    if config['POT']['MultiArith']:
        print('MultiArith\n')
        pot_acc['MultiArith'] = get_accuracy_pot(config['POT']['MultiArith'])
        
    # Select-Sample 
    ss_acc = {'GSM8K':-1, 'SVAMP':-1, 'ASDiv':-1, 'MultiArith':-1}
    strat_dict = {'GSM8K':{}, 'SVAMP':{}, 'ASDiv':{}, 'MultiArith':{}}
    print("\nSS Predictions\n")
    if config['SELSAMP']['GSM8K']:
        only_one_strat = 'SELSAMP'
        num_strats += 1
        print("GSM8K\n")
        ss_acc['GSM8K'], strat_dict['GSM8K'] = get_accuracy_ss(config['SELSAMP']['GSM8K'])
    if config['SELSAMP']['SVAMP']:
        print("SVAMP\n")
        ss_acc['SVAMP'], strat_dict['SVAMP'] = get_accuracy_ss(config['SELSAMP']['SVAMP'])
    if config['SELSAMP']['ASDiv']:
        print("ASDiv\n")
        ss_acc['ASDiv'], strat_dict['ASDiv'] = get_accuracy_ss(config['SELSAMP']['ASDiv'])
    if config['SELSAMP']['MultiArith']:
        print('MultiArith\n')
        ss_acc['MultiArith'], strat_dict['MultiArith'] = get_accuracy_ss(config['SELSAMP']['MultiArith'])
    
    # Select-Sample CoT Biased 
    ss_cot_acc = {'GSM8K':-1, 'SVAMP':-1, 'ASDiv':-1, 'MultiArith':-1}
    strat_cot_dict = {'GSM8K':{}, 'SVAMP':{}, 'ASDiv':{}, 'MultiArith':{}}
    print("\nSS COT Predictions\n")
    if config['SELSAMP_COT_BIASED']['GSM8K']:
        only_one_strat = 'SELSAMP_COT_BIASED'
        num_strats += 1
        print("GSM8K\n")
        ss_cot_acc['GSM8K'], strat_cot_dict['GSM8K'] = get_accuracy_ss(config['SELSAMP_COT_BIASED']['GSM8K'])
    if config['SELSAMP_COT_BIASED']['SVAMP']:
        print("SVAMP\n")
        ss_cot_acc['SVAMP'], strat_cot_dict['SVAMP'] = get_accuracy_ss(config['SELSAMP_COT_BIASED']['SVAMP'])
    if config['SELSAMP_COT_BIASED']['ASDiv']:
        print("ASDiv\n")
        ss_cot_acc['ASDiv'], strat_cot_dict['ASDiv'] = get_accuracy_ss(config['SELSAMP_COT_BIASED']['ASDiv'])
    if config['SELSAMP_COT_BIASED']['MultiArith']:
        print('MultiArith\n')
        ss_cot_acc['MultiArith'], strat_cot_dict['MultiArith'] = get_accuracy_ss(config['SELSAMP_COT_BIASED']['MultiArith'])
    
    # Select-Sample PoT Biased
    ss_pot_acc = {'GSM8K':-1, 'SVAMP':-1, 'ASDiv':-1, 'MultiArith':-1}
    strat_pot_dict = {'GSM8K':{}, 'SVAMP':{}, 'ASDiv':{}, 'MultiArith':{}}
    print("\nSS POT Predictions\n")
    if config['SELSAMP_POT_BIASED']['GSM8K']:
        only_one_strat = 'SELSAMP_POT_BIASED'
        num_strats += 1
        print("GSM8K\n")
        ss_pot_acc['GSM8K'], strat_pot_dict['GSM8K'] = get_accuracy_ss(config['SELSAMP_POT_BIASED']['GSM8K'])
    if config['SELSAMP_POT_BIASED']['SVAMP']:
        print("SVAMP\n")
        ss_pot_acc['SVAMP'], strat_pot_dict['SVAMP'] = get_accuracy_ss(config['SELSAMP_POT_BIASED']['SVAMP'])
    if config['SELSAMP_POT_BIASED']['ASDiv']:
        print("ASDiv\n")
        ss_pot_acc['ASDiv'], strat_pot_dict['ASDiv'] = get_accuracy_ss(config['SELSAMP_POT_BIASED']['ASDiv'])
    if config['SELSAMP_POT_BIASED']['MultiArith']:
        print('MultiArith\n')
        ss_pot_acc['MultiArith'], strat_pot_dict['MultiArith'] = get_accuracy_ss(config['SELSAMP_POT_BIASED']['MultiArith'])

    
    # Select-Sample L2M Biased
    ss_l2m_acc = {'GSM8K':-1, 'SVAMP':-1, 'ASDiv':-1, 'MultiArith':-1}
    strat_l2m_dict = {'GSM8K':{}, 'SVAMP':{}, 'ASDiv':{}, 'MultiArith':{}}
    print("\nSS L2M Predictions\n")
    if config['SELSAMP_L2M_BIASED']['GSM8K']:
        only_one_strat = 'SELSAMP_L2M_BIASED'
        num_strats += 1
        print("GSM8K\n")
        ss_l2m_acc['GSM8K'], strat_l2m_dict['GSM8K'] = get_accuracy_ss(config['SELSAMP_L2M_BIASED']['GSM8K'])
    if config['SELSAMP_L2M_BIASED']['SVAMP']:
        print("SVAMP\n")
        ss_l2m_acc['SVAMP'], strat_l2m_dict['SVAMP'] = get_accuracy_ss(config['SELSAMP_L2M_BIASED']['SVAMP'])
    if config['SELSAMP_L2M_BIASED']['ASDiv']:
        print("ASDiv\n")
        ss_l2m_acc['ASDiv'], strat_l2m_dict['ASDiv'] = get_accuracy_ss(config['SELSAMP_L2M_BIASED']['ASDiv'])
    if config['SELSAMP_L2M_BIASED']['MultiArith']:
        print('MultiArith\n')
        ss_l2m_acc['MultiArith'], strat_l2m_dict['MultiArith'] = get_accuracy_ss(config['SELSAMP_L2M_BIASED']['MultiArith'])


    # Select-Sample Weighted 
    ss_wtd_acc = {'GSM8K':-1, 'SVAMP':-1, 'ASDiv':-1, 'MultiArith':-1}
    strat_wtd_dict = {'GSM8K':{}, 'SVAMP':{}, 'ASDiv':{}, 'MultiArith':{}}
    print("\nSS Weighted Predictions\n")
    if config['SELSAMP_WEIGHTED']['GSM8K']:
        num_strats += 1
        only_one_strat = 'SELSAMP_WEIGHTED'
        print("GSM8K\n")
        ss_wtd_acc['GSM8K'], strat_wtd_dict['GSM8K'] = get_accuracy_ss(config['SELSAMP_WEIGHTED']['GSM8K'])
    if config['SELSAMP_WEIGHTED']['SVAMP']:
        print("SVAMP\n")
        ss_wtd_acc['SVAMP'], strat_wtd_dict['SVAMP'] = get_accuracy_ss(config['SELSAMP_WEIGHTED']['SVAMP'])
    if config['SELSAMP_WEIGHTED']['ASDiv']:
        print("ASDiv\n")
        ss_wtd_acc['ASDiv'], strat_wtd_dict['ASDiv'] = get_accuracy_ss(config['SELSAMP_WEIGHTED']['ASDiv'])
    if config['SELSAMP_WEIGHTED']['MultiArith']:
        print('MultiArith\n')
        ss_wtd_acc['MultiArith'], strat_wtd_dict['MultiArith'] = get_accuracy_ss(config['SELSAMP_WEIGHTED']['MultiArith'])


    acc_data = {
    'COT': cot_acc,
    'L2M': l2m_acc,
    'POT': pot_acc,
    'SelSamp': ss_acc,
    'SelSamp-CoT-Biased': ss_cot_acc,
    'SelSamp-PoT-Biased': ss_pot_acc, 
    'SelSamp-L2M-Biased': ss_l2m_acc,
    'Selsamp-Weighted':ss_wtd_acc}
    
    df_acc = pd.DataFrame(acc_data).T 
    print("Accuracies: \n", df_acc)
    
    if config['SELSAMP']['GSM8K']:
        df_strat_ss = pd.DataFrame(strat_dict)
        print("Strategy Distribution :\n", df_strat_ss)
    else:
        df_strat_ss = pd.DataFrame()
    
    if config['SELSAMP_COT_BIASED']['GSM8K']:
        df_strat_ss_cot = pd.DataFrame(strat_cot_dict)
        print("Strategy Distribution CoT :\n", df_strat_ss_cot)
    else:
        df_strat_ss_cot = pd.DataFrame()

    if config['SELSAMP_POT_BIASED']['GSM8K']:
        df_strat_ss_pot = pd.DataFrame(strat_pot_dict)
        print("Strategy Distribution PoT :\n", df_strat_ss_pot)
    else:
        df_strat_ss_pot = pd.DataFrame()

    if config['SELSAMP_L2M_BIASED']['GSM8K']:
        df_strat_ss_l2m = pd.DataFrame(strat_l2m_dict)
        print("Strategy Distribution L2M :\n", df_strat_ss_l2m)
    else:
        df_strat_ss_l2m = pd.DataFrame()

    if config['SELSAMP_WEIGHTED']['GSM8K']:
        df_strat_ss_wtd = pd.DataFrame(strat_wtd_dict)
        print("Strategy Distribution Weighted :\n", df_strat_ss_wtd)
    else:
        df_strat_ss_wtd = pd.DataFrame()

    # Save All 
    model_dir = config['MODEL_DIR']

    if num_strats==1:
        chk = config[only_one_strat]['GSM8K'].split('/')[-1].replace(".csv","").split("-")[-1]
        df_acc.to_csv(os.path.join(model_dir, 'test_accs_{x}.csv'.format(x='chk-'+chk)))
        df_strat_ss.to_csv(os.path.join(model_dir, 'strat_ss_dist_{x}.csv'.format(x='chk-'+chk)))
    else:
        df_acc.to_csv(os.path.join(model_dir, 'test_accs.csv'))
        df_strat_ss.to_csv(os.path.join(model_dir, 'strat_ss_dist.csv'))
    df_strat_ss_cot.to_csv(os.path.join(model_dir, 'strat_ss_cot_dist.csv'))
    df_strat_ss_pot.to_csv(os.path.join(model_dir, 'strat_ss_pot_dist.csv'))
    df_strat_ss_l2m.to_csv(os.path.join(model_dir, 'strat_ss_l2m_dist.csv'))
    df_strat_ss_wtd.to_csv(os.path.join(model_dir, 'strat_ss_wtd_dist.csv'))

    return df_acc, df_strat_ss, df_strat_ss_cot, df_strat_ss_pot, df_strat_ss_l2m, df_strat_ss_wtd


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Results - Gemma GSM8K Fine-tuned")
    parser.add_argument("config_path", type=str, help="Path to the configuration file")
    args = parser.parse_args()
    df_acc, df_strat_ss, df_strat_ss_cot, df_strat_ss_pot, df_strat_ss_l2m, df_strat_ss_wtd = run(args.config_path)
