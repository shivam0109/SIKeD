import argparse
import json
import re
import jsonlines
from fraction import Fraction
from vllm import LLM, SamplingParams
import sys
import pandas as pd
from tqdm import tqdm
import torch

MAX_INT = sys.maxsize
RAY_memory_monitor_refresh_ms=0

# read data from the file
def get_dataset(PATH):
    with open(PATH, 'r') as dataset:
        data_list = list(dataset)
        for line in data_list:
            problem = json.loads(line)
            yield problem

#extract prediction
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

def read_prompt(PROMPT_PATH):
    with open(PROMPT_PATH, "r") as f:
        prompt = f.read()
    return prompt

# save results to a file
def store_result(out, problem, output, first_steps="", first_step=False):
    if first_step:
        problem['first_step'] = first_steps
    problem['predicted_label'] = output
    out.write(json.dumps(problem, ensure_ascii=False) + '\n')

def gsm8k_test(model, mmlu, data_path, output_path, first_step_path, first_step, tensor_parallel_size, prompt_path):
    stop_tokens = ["Input", "Input:", "<eot_id>", "<|im_end|>"]
    sampling_params = SamplingParams(temperature=0, n=1, top_p=0.95, max_tokens=512, stop=stop_tokens)
    llm = LLM(model=model, tensor_parallel_size=tensor_parallel_size, trust_remote_code=True, max_model_len=4000)
    instruction = "Below is an instruction that describes a task.\n Write a response that appropriately completes the request.\n\n"
    prompt = read_prompt(prompt_path)
    if first_step:
        print ("First step!")
        # first_steps = [sample['prediction'][0][-1].split("The answer is ")[0] for sample in get_dataset(first_step_path)]
        first_steps = [sample['completion'].split(".")[0] for sample in get_dataset(first_step_path)]
        # 
        # create choices with A, B, C, D
        if mmlu:
            print ("MMLU task!")
            choices = ["A", "B", "C", "D"]
            all_prompt_formatted = [f"{prompt} \n\n### Input:\n {sample['query']}\nOptions: {''.join([f'{choice}. {option} ' for choice, option in zip(choices, sample['choices'])])}### Response:\n {fs} " for sample, fs in zip(get_dataset(data_path), first_steps)]
        else:
            all_prompt_formatted = [f"{prompt} \n\n### Input:\n {sample['query']}\n### Response:\n {fs} " for sample, fs in zip(get_dataset(data_path), first_steps)]
        # all_prompt_formatted = [f"{instruction} ### Instruction: Solve the math word problem in step by step manner. \n\n### Input:\nQuestion: {sample['query']}\n### Response: {fs} " for sample, fs in zip(get_dataset(data_path), first_steps)]
    else:
        print ("No first step!")
        if mmlu:
            print ("MMLU task!")
            choices = ["A", "B", "C", "D"]
            all_prompt_formatted = [f"{prompt} \n\n### Input:\n {sample['query']}\nOptions: {''.join([f'{choice}. {option} ' for choice, option in zip(choices, sample['choices'])])}### Response:\n " for sample in get_dataset(data_path)]
        else:
            all_prompt_formatted = [f"{prompt}\n### Input:\n{sample['query']}\n### Response:\n" for sample in get_dataset(data_path)]
        # all_prompt_formatted = [f"{instruction} ### Instruction: Solve the math word problem in step by step manner. \n\n### Input:\nQuestion: {sample['query']}\n### Response: " for sample in get_dataset(data_path)]

    outputs = llm.generate(all_prompt_formatted, sampling_params)
    # extract predictions
    print (len(outputs))
    # create a completion list with size = number of samples
    completions = []
    for output in outputs:
        temp = []
        for i in range(len(output.outputs)):
            temp.append(output.outputs[i].text)
        completions.append(temp)
    print (len(completions))

    if first_step:
        with open(output_path, "w") as out:
            for sample, output, fs in zip(get_dataset(data_path), completions, first_steps):
                store_result(out, sample, output, fs, first_step=True)
    else:
        with open(output_path, "w") as out:
            for sample, output_ in zip(get_dataset(data_path), completions):
                store_result(out, sample, output_)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="/home/shri/git/mygit/see-then-act/models/shivam-qwen-point5-merged/l2m")  # model path
    parser.add_argument("--mmlu", type=bool, default=False)  # MMLU task or not
    parser.add_argument("--data_file", type=str, default='./data/multiarith.jsonl')  # data path
    parser.add_argument("--output_path", type=str, default="./results/shivam-qwen-point5/multiarith_test_l2m.jsonl")  # tensor_parallel_size
    parser.add_argument("--first_step_path", type=str, default="data/svamp_llama-70b.jsonl")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)  # tensor_parallel_size
    parser.add_argument("--first_step", type=bool, default=False)
    parser.add_argument("--prompt_path", type=str, default="./data/prompts/shivam-prompt-selsam.txt")

    return parser.parse_args()
    
if __name__ == "__main__":
    args = parse_args()
    gsm8k_test(model=args.model, mmlu=args.mmlu, data_path=args.data_file, output_path=args.output_path, first_step_path=args.first_step_path, first_step=args.first_step, tensor_parallel_size=args.tensor_parallel_size, prompt_path=args.prompt_path)
