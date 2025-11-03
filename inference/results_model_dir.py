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
import xml.etree.ElementTree as ET

# Note: This path should be configured to point to your AsDiV dataset XML file
# You may need to download it from: https://github.com/chaochun/nlu-asdiv-dataset
ASDIV_PATH = None  # Set this to the path to your ASDiv.xml file, e.g., '/path/to/ASDiv.xml'

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

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
def load_asdiv():
    if ASDIV_PATH is None:
        raise ValueError("ASDIV_PATH is not set. Please configure the path to ASDiv.xml in results_model_dir.py")
    tree = ET.parse(ASDIV_PATH)
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
    if 'asdiv' in df_path:
        try:
            df['answer'] = df['answer'].apply(float)
        except:
            df_asdiv = load_asdiv()
            df.drop(columns=['answer'], inplace=True)
            df = df.merge(df_asdiv[['id','answer']], on='id', how='inner')
            assert(df.shape[0]==2305)
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
        if len(ans) > 0:
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
    if 'asdiv' in df_path:
        try:
            df['answer'] = df['answer'].apply(float)
        except:
            df_asdiv = load_asdiv()
            df.drop(columns=['answer'], inplace=True)
            df = df.merge(df_asdiv[['id','answer']], on='id', how='inner')
            assert(df.shape[0]==2305)
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
    if 'asdiv' in df_path:
        try:
            df['answer'] = df['answer'].apply(float)
        except:
            df_asdiv = load_asdiv()
            df.drop(columns=['answer'], inplace=True)
            df = df.merge(df_asdiv[['id','answer']], on='id', how='inner')
            assert(df.shape[0]==2305)

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


def run(model_dir):
    # Get all checkpoint names 
    checkpoints = [x for x in os.listdir(model_dir) if x.startswith('checkpoint')] + ['final_model']
    
    # Create accuracy table 
    accuracy_table = {'checkpoint':[], 'asdiv':[], 'gsm8k':[], 'svamp':[], 'multiarith':[]}

    # Loop over all checkpoints 
    for chk in checkpoints: 
        accuracy_table['checkpoint'].append(chk)
        asdiv_pred_path = os.path.join(model_dir, 'asdiv_predictions_{x}.csv'.format(x=chk))
        gsm8k_pred_path = os.path.join(model_dir, 'gsm8k_predictions_{x}.csv'.format(x=chk))
        svamp_pred_path = os.path.join(model_dir, 'svamp_predictions_{x}.csv'.format(x=chk))
        multi_pred_path = os.path.join(model_dir, 'multiarith_predictions_{x}.csv'.format(x=chk))
        
        if 'selsamp' in model_dir:
            strat_dict = {'GSM8K':{}, 'SVAMP':{}, 'ASDiv':{}, 'MultiArith':{}}
            if os.path.exists(asdiv_pred_path):
                acc_asdiv, strat_dict['ASDiv'] = get_accuracy_ss(asdiv_pred_path)
                accuracy_table['asdiv'].append(acc_asdiv)
            else:
                accuracy_table['asdiv'].append(-1)

            if os.path.exists(gsm8k_pred_path):
                acc_gsm8k, strat_dict['GSM8K'] = get_accuracy_ss(gsm8k_pred_path)
                accuracy_table['gsm8k'].append(acc_gsm8k)
            else:
                accuracy_table['gsm8k'].append(-1)

            if os.path.exists(svamp_pred_path):
                acc_svamp, strat_dict['SVAMP'] = get_accuracy_ss(svamp_pred_path)
                accuracy_table['svamp'].append(acc_svamp)
            else:
                accuracy_table['svamp'].append(-1)
            
            if os.path.exists(multi_pred_path):
                acc_ma, strat_dict['MultiArith'] = get_accuracy_ss(multi_pred_path)
                accuracy_table['multiarith'].append(acc_ma)
            else:
                accuracy_table['multiarith'].append(-1)

            # Save strategy distribution 
            pd.DataFrame(strat_dict).to_csv(os.path.join(model_dir,'strat-dist-{x}.csv'.format(x=chk)), index=True)

        elif 'pot' in model_dir:
            if os.path.exists(asdiv_pred_path):
                acc_asdiv = get_accuracy_pot(asdiv_pred_path)
                accuracy_table['asdiv'].append(acc_asdiv)
            else:
                accuracy_table['asdiv'].append(-1)

            if os.path.exists(gsm8k_pred_path):
                acc_gsm8k = get_accuracy_pot(gsm8k_pred_path)
                accuracy_table['gsm8k'].append(acc_gsm8k)
            else:
                accuracy_table['gsm8k'].append(-1)

            if os.path.exists(svamp_pred_path):
                acc_svamp = get_accuracy_pot(svamp_pred_path)
                accuracy_table['svamp'].append(acc_svamp)
            else:
                accuracy_table['svamp'].append(-1)
            
            if os.path.exists(multi_pred_path):
                acc_ma = get_accuracy_pot(multi_pred_path)
                accuracy_table['multiarith'].append(acc_ma)
            else:
                accuracy_table['multiarith'].append(-1)

        else:
            if os.path.exists(asdiv_pred_path):
                acc_asdiv = get_accuracy(asdiv_pred_path)
                accuracy_table['asdiv'].append(acc_asdiv)
            else:
                accuracy_table['asdiv'].append(-1)

            if os.path.exists(gsm8k_pred_path):
                acc_gsm8k = get_accuracy(gsm8k_pred_path)
                accuracy_table['gsm8k'].append(acc_gsm8k)
            else:
                accuracy_table['gsm8k'].append(-1)

            if os.path.exists(svamp_pred_path):
                acc_svamp = get_accuracy(svamp_pred_path)
                accuracy_table['svamp'].append(acc_svamp)
            else:
                accuracy_table['svamp'].append(-1)
            
            if os.path.exists(multi_pred_path):
                acc_ma = get_accuracy(multi_pred_path)
                accuracy_table['multiarith'].append(acc_ma)
            else:
                accuracy_table['multiarith'].append(-1)
            

    # Sort and save accuracy tables
    df_acc =  pd.DataFrame(accuracy_table)
    df_acc_gsm8k = df_acc.sort_values(by='gsm8k', key=lambda x: x.apply(lambda y: y[1]), ascending=False)
    df_acc_svamp = df_acc.sort_values(by='svamp', key=lambda x: x.apply(lambda y: y[1]), ascending=False)
    df_acc_asdiv = df_acc.sort_values(by='asdiv', key=lambda x: x.apply(lambda y: y[1]), ascending=False)
    df_acc_ma = df_acc.sort_values(by='multiarith', key=lambda x: x.apply(lambda y: y[1]), ascending=False)
    
    df_acc_gsm8k.to_csv(os.path.join(model_dir, 'test_accs_all_checkpoints_gsm8k_sorted.csv'), index=False)
    df_acc_svamp.to_csv(os.path.join(model_dir, 'test_accs_all_checkpoints_svamp_sorted.csv'), index=False)
    df_acc_asdiv.to_csv(os.path.join(model_dir, 'test_accs_all_checkpoints_asdiv_sorted.csv'), index=False)
    df_acc_ma.to_csv(os.path.join(model_dir, 'test_accs_all_checkpoints_multiarith_sorted.csv'), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Results Script")
    parser.add_argument("model_dir", type=str, help="Path to the model directory")
    args = parser.parse_args()
    run(args.model_dir)
