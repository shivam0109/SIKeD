from asyncore import read
import sys
from io import StringIO
import os
import json 
import numpy as np

# GLOBAL VARS 
PREFIX_FRN = "prefix.txt"
# Note: NLSL_PATH and NONL_PATH are legacy paths that are likely not used anymore
NLSL_PATH = None  # "/path/to/llama-70B_NL+SL/predictions.jsonl"
NONL_PATH = None  # "/path/to/llama-70B_noNL/predictions.jsonl"

def solve_mwp(completion, prefix_frn=PREFIX_FRN):
    
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

def read_data_helper(path):
    with open(path, 'r') as json_file:
        data = [json.loads(line) for line in json_file]
    return data 


def get_answers(data):
    answers = []
    for i in range(len(data)):
        completion = data[i]['completion']
        answer = solve_mwp(completion)
        answers.append(answer)
    return answers 

def check_answers(answers):
    count = 0 
    for i in range(len(answers)):
        if answers[i]=="[invalid]":
            count += 1
    perc_count = np.round(100*count/len(answers), 2)
    print("[invalid] answers for {x} data points ; {y} percentage".format(x=count, y=perc_count))
    return count 


if __name__ == "__main__":
    nlsl_data = read_data_helper(NLSL_PATH)
    nonl_data = read_data_helper(NONL_PATH)
    nlsl_answers = get_answers(nlsl_data)
    nonl_answers = get_answers(nonl_data)
    print("Checking NoNL Data")
    check_answers(nonl_answers)
    print("Checking NLSL Data")
    check_answers(nlsl_answers)
