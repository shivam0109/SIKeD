import pandas as pd 
import datasets 
import json 
import os 
import numpy as np 
import signal 
import re 
from functools import reduce
import ast 
import string 
translator = str.maketrans('', '', string.punctuation)

# COT_PATH = "/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/predictions_test_cot.jsonl"
# POT_PATH = "/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/predictions_test_pot.jsonl"
# L2M_PATH = "/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/predictions_test_l2m.jsonl"

# COT_PATH = "/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/gemma-epoch3/df_forced_preds_cot_test.csv"
# POT_PATH = "/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/gemma-epoch3/df_forced_preds_pot_test.csv"
# L2M_PATH = "/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/gemma-epoch3/df_forced_preds_l2m_test.csv"

# COT_PATH = "/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/gemma-7b-It/epoch2/df_forced_preds_cot_test.csv"
# POT_PATH = "/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/gemma-7b-It/epoch2/df_forced_preds_pot_test.csv"
# L2M_PATH = "/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/gemma-7b-It/epoch2/df_forced_preds_l2m_test.csv"

# COT_PATH = "/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/qwen2-1.5b-epoch2/df_forced_preds_cot_test.csv"
# POT_PATH = "/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/qwen2-1.5b-epoch2/df_forced_preds_pot_test.csv"
# L2M_PATH = "/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/qwen2-1.5b-epoch2/df_forced_preds_l2m_test.csv"

# COT_PATH = "/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/qwen2-0.5b-epoch2/df_forced_preds_cot_test.csv"
# POT_PATH = "/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/qwen2-0.5b-epoch2/df_forced_preds_pot_test.csv"
# L2M_PATH = "/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/qwen2-0.5b-epoch2/df_forced_preds_l2m_test.csv"

# COT_PATH = "/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/qwen2-0.5b-epoch3/df_forced_preds_cot_test.csv"
# POT_PATH = "/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/qwen2-0.5b-epoch3/df_forced_preds_pot_test.csv"
# L2M_PATH = "/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/qwen2-0.5b-epoch3/df_forced_preds_l2m_test.csv"

COT_PATH = "/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/qwen2-1.5b-epoch3/df_forced_preds_cot_test.csv"
POT_PATH = "/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/qwen2-1.5b-epoch3/df_forced_preds_pot_test.csv"
L2M_PATH = "/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/qwen2-1.5b-epoch3/df_forced_preds_l2m_test.csv"


def read_data(file_path, strategy, pred_colname):
    if file_path.endswith("jsonl"):
        data = []
        
        # Open the file in read mode
        with open(file_path, 'r', encoding='utf-8') as file:
            # Read each line in the file
            for line in file:
                # Parse the line as JSON and add to the list
                data.append(json.loads(line))
        
        df = pd.DataFrame(data)

    if file_path.endswith("csv"):
        df = pd.read_csv(file_path)

    if pred_colname == 'predictions':
    # if isinstance(df[pred_colname].iloc[0], str):
        df[pred_colname] = df[pred_colname].apply(lambda x: ast.literal_eval(x))

    df = df.reset_index(drop=True)
    # Replace '[cot]:', '[pot]:', '[l2m]:' tokens and select the first element 
    df['predictions_'+strategy] = df[pred_colname].apply(lambda x: x[0].replace('[cot]:','').replace('[pot]:','').replace('[l2m]:','').strip())

    if 'question' not in df.columns:
        df.rename(columns={'query':'question'}, inplace=True)
    print("Input Shape: ", df.shape)
    if 'response' in df.columns:
        return df[['question','response','predictions_'+strategy]].copy()
    if 'answer' in df.columns:
        return df[['question','answer','predictions_'+strategy]].copy()
    

def check_l2m_validity(reasoning_chain):
    reasoning_chain = reasoning_chain.lower()
    cleaned_rc = reasoning_chain.translate(translator)
    return 'subquestion' in cleaned_rc

def select_l2m_rc(df_l2m, answer_col):
    print("Checking L2M Reasoning Chains")
    print("Intiial shape of L2M Reasoning Chain: ", df_l2m.shape)
    df_l2m['is_l2m'] = df_l2m[answer_col].apply(lambda x: check_l2m_validity(x))
    df_l2m_rc = df_l2m[df_l2m['is_l2m']==True].copy()
    df_l2m_rc.drop(columns=['is_l2m'], inplace=True)
    df_l2m_rc.reset_index(drop=True, inplace=True)
    print("Shape after removing noise in L2M Reasoning Chains: ", df_l2m_rc.shape)
    return df_l2m_rc


## Function to get Predicted Answers in Numeric Format
def get_pred_answers(pred):
    pred = pred.lower().replace("<pad>","").replace("<eos>","")
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

def extract_pred(pred_sample):
    # val = sample.split("The answer is ")[-1]
    pattern = r'[$]?[-+]?\d+(?:\.\d+)?(?:,\d+)*[$]?'
    matches = re.findall(pattern, pred_sample)
    if  matches != []:
        final_pred = str(matches[-1].replace(",", "").replace(" ", "").replace("\n", "").replace("$", "").replace("x", ""))
    else:
        final_pred = str(0)
        print ("No answer found!")
    return final_pred


# Function to get accuracy of predictions - for CoT and L2M
def get_accuracy(df, strategy):
    try:
        df['answer'] = df['answer'].apply(float)
    except:
        df['answer'] = pd.to_numeric(df['answer'], errors='coerce')
        print("Removed {x} rows with non-float true answers".format(x=df['answer'].isnull().sum()))
        df = df.dropna()
    
    df['predicted_num_answer'] = df['predictions_'+strategy].apply(lambda x: get_pred_answers(x))
    not_found = (df['predicted_num_answer']==-2048).sum()
    print("Predicted answer not found for {x} samples".format(x=not_found))
    df['is_correct'] = df.apply(lambda row: float(row['answer'])==float(row['predicted_num_answer']), axis=1)
    acc = (df['is_correct'].sum(), np.round(100*df['is_correct'].sum()/1319, 2))
    return df, acc 

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
    completions = [x.replace('<pad>','').replace('<eos>','').split('Response:')[-1] for x in predictions]
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


def get_accuracy_pot(df):
    try:
        df['answer'] = df['answer'].apply(float)
    except:
        df['answer'] = pd.to_numeric(df['answer'], errors='coerce')
        print("Removed {x} rows with non-float true answers".format(x=df['answer'].isnull().sum()))
        df = df.dropna()
    predictions = df['predictions_pot'].tolist()
    # Format predictions 
    formatted_predictions = format_predictions(predictions)
    # Get Numeric answers from formatted predictions 
    predicted_answers = [extract_pred_math_solver(formatted_prediction) for formatted_prediction in formatted_predictions]
    df['predicted_num_answer'] = predicted_answers
    not_found = predicted_answers.count(-2048)
    print("Predicted answer not found for {x} samples".format(x=not_found))
    df['is_correct'] = df.apply(lambda row: is_correct_helper(row['answer'], row['predicted_num_answer']), axis=1)
    acc = (df['is_correct'].sum(), np.round(100*df['is_correct'].sum()/1319, 2))
    return df, acc 


def get_true_num_answer(df, colname='response'):
    if 'answer' in df.columns:
        df['answer'] = df['answer'].apply(float)
    else:
        df['answer'] = df[colname].apply(lambda x: float(x.split("\n")[-1].replace("#",'').replace(",", "").replace(" ", "").replace("\n", "").replace("$", "").replace("x", "").strip()))
    return df 

def merge_dfs(df_cot, df_pot, df_l2m):
    dfs = [df_cot, df_pot, df_l2m]
    df_merged = reduce(lambda left,right: pd.merge(left,right,on=['question']), dfs)
    print("Merged DF Shape: ", df_merged.shape)
    print("Null Values:\n")
    print(df_merged.isnull().sum())
    return df_merged 

def run():
    # import pdb ; pdb.set_trace()
    # df_cot = read_data(COT_PATH, 'cot', 'predicted_label')
    df_cot = read_data(COT_PATH, 'cot', 'predictions')
    df_cot = get_true_num_answer(df_cot)
    df_cot, acc_cot = get_accuracy(df_cot, 'cot')
    df_cot.rename(columns={'predicted_num_answer':'predicted_num_answer_cot', 
                           'is_correct':'is_correct_cot'}, inplace=True)

    # df_pot = read_data(POT_PATH, 'pot', 'predicted_label')
    df_pot = read_data(POT_PATH, 'pot', 'predictions')
    df_pot = get_true_num_answer(df_pot)
    df_pot, acc_pot = get_accuracy_pot(df_pot)
    df_pot.rename(columns={'predicted_num_answer':'predicted_num_answer_pot', 
                           'is_correct':'is_correct_pot'}, inplace=True)

    
    df_l2m = read_data(L2M_PATH, 'l2m', 'predictions')
    df_l2m = select_l2m_rc(df_l2m, 'predictions_l2m')
    # df_l2m = read_data(L2M_PATH, 'l2m', 'predicted_label')
    df_l2m = get_true_num_answer(df_l2m)
    df_l2m, acc_l2m = get_accuracy(df_l2m, 'l2m')
    df_l2m.rename(columns={'predicted_num_answer':'predicted_num_answer_l2m', 
                           'is_correct':'is_correct_l2m'}, inplace=True)
    
    df_merged = merge_dfs(df_cot, df_pot, df_l2m)
    df_merged['any_is_correct'] = df_merged['is_correct_cot'] | df_merged['is_correct_pot'] | df_merged['is_correct_l2m']
    print("CoT Accuracy: ", acc_cot)
    print("PoT Accuracy: ", acc_pot)
    print("L2M Accuracy: ", acc_l2m)

    print("Any-is-correct Accuracy: \n")
    print(np.round(100*df_merged['any_is_correct'].sum()/1319, 2))


if __name__ == "__main__":
    run()








