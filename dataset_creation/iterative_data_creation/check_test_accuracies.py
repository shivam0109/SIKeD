import json 
import numpy as np 
import pandas as pd 
from datasets import load_dataset
import signal 
import argparse

def read_data(file_path):
    # Read Jsonl
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    
    # Convert to DF 
    df = pd.DataFrame(data)
    df['query'] = df['query'].apply(lambda x: x.strip())
    df.rename(columns={'query':'question'}, inplace=True)

    # Merge with gsm8k
    df_gsm8k = load_dataset('gsm8k','main',split='test').to_pandas() 
    df_gsm8k['question'] = df_gsm8k['question'].apply(lambda x: x.strip())

    df_merged = df.merge(df_gsm8k, on=['question'], how='inner')
    print("Shape of merged data: ", df_merged.shape)
    return df_merged 

def get_true_answer(df):
    df['num_true_ans'] = df['answer'].apply(lambda ans: ans.split('\n')[-1].replace('#','').replace(",", "").replace(" ", "").replace("\n", "").replace("$", "").replace("x", "").strip())
    df['num_true_ans'] = df['num_true_ans'].apply(lambda x: float(x))

## Function to get Predicted Answers in Numeric Format
## For CoT and L2M 
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

# Get Predicted Answer
def get_predicted_answer(df, is_pot=False):
    if is_pot:
        df['num_pred_ans'] = df['predicted_label'].apply(lambda x: extract_pred_math_solver(x[0]))
    else:
        df['num_pred_ans'] = df['predicted_label'].apply(lambda x: get_pred_answers(x[0]))


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


# Get Accuracy 
def get_accuracy(df):
    print("No answer found for {x} samples".format(x=df['num_pred_ans'].tolist().count(-2048) + df['num_pred_ans'].tolist().count('[invalid]')))
    df['is_correct'] = df.apply(lambda row: is_correct_helper(row['num_true_ans'], row['num_pred_ans']), axis=1)
    acc = (df['is_correct'].sum(), np.round(100*df['is_correct'].sum()/df.shape[0], 2))
    print("Accuracy: ", acc)


def run(file_path):
    df = read_data(file_path)
    get_true_answer(df)
    if 'pot' in file_path:
        get_predicted_answer(df, is_pot=True)
    else:
        get_predicted_answer(df)
    get_accuracy(df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Forced Test Accuracies")
    parser.add_argument("file_path", type=str, help="Path to the input file")
    args = parser.parse_args()
    file_path = args.file_path
    run(file_path) 
