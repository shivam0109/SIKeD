"""
Code for knowledge distillation using LoRA 
"""
import sys
for path in sys.path:
    print(path)
import argparse
import yaml 
from unsloth import FastLanguageModel
import torch
import pandas as pd 
import random 
import datasets
from trl import SFTTrainer
from transformers import TrainingArguments, HfArgumentParser
import wandb 
import os 


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


# Function to get Train and Val DataFrames - No Common IDs 
def split_train_val(df, ratio=0.8):
    ids = list(set(df['id'].tolist()))
    print("Number of unique IDs: ", len(ids))
    random.seed(SEED)
    random.shuffle(ids)
    
    ntrain = int(ratio*len(ids))
    train_ids = ids[:ntrain]
    val_ids = ids[ntrain:]
    
    df_train = df[df['id'].isin(train_ids)].copy()
    df_val = df[df['id'].isin(val_ids)].copy()
    
    print("Train shape: ", df_train.shape)
    print("Val shape: ", df_val.shape)
    print("# IDs in Train: ", len(df_train['id'].unique()))
    print("# IDs in Validation: ", len(df_val['id'].unique()))
    print("Train-Val distribution: ", df_train.shape[0]/(df_train.shape[0] + df_val.shape[0]))
    
    return df_train, df_val 


# Function to arrange dataframe such that the first n rows correspond to first n responses, and so on. 
# The idea is that the model sees all IDs first, instead of all responses from an ID 
def arrange_df(df, id_col='id', ques_col='question',response_col='response'):
    df = df.sort_values(by=[id_col])
    
    # Group dataframe by 'id' column
    grouped = df.groupby(id_col)
    
    # Initialize an empty list to store the arranged data
    arranged_data = []
    
    # Get unique IDs and the maximum number of responses for any ID
    unique_ids = df[id_col].unique()
    max_responses = grouped.size().max()
    
    # Iterate over the number of responses (max_responses)
    for i in range(max_responses):
        # Iterate over unique IDs
        for id_ in unique_ids:
            # Get all responses and questions for the current ID
            responses = grouped.get_group(id_)[response_col].tolist()
            questions = grouped.get_group(id_)[ques_col].tolist()
            # Append the ID, question, and response if available, else append None
            if i < len(responses):
                arranged_data.append({'id': id_, 'question': questions[i], 'response': responses[i]})
            else:
                arranged_data.append({'id': id_, 'question': None, 'response': None})
    
    # Create a new DataFrame from the arranged data
    arranged_df = pd.DataFrame(arranged_data)
    arranged_df = arranged_df.dropna(subset=['response'])
    arranged_df.reset_index(drop=True, inplace=True)
    print(arranged_df.head(10))
    
    assert(arranged_df.shape[0] == df.shape[0])
    return arranged_df


# Prepare Training and Validation Data 
# Inputs: 
# col_name: Dataframe column containing LLM responses. 
# path: data path 
# multiple_strategies: use multiple strategies during training - True only for select_sample  
# arrange_train: arrange training data such that all questions are seen first during training. 
#                Only applicable when multiple generations of same question are present. 
# remove_train_duplicates: remove duplicate questions in training data. Only applicable when multiple generations of same question are present. 
# remove_val_duplicates: remove duplicate questions in validation data. Only applicable when multiple generations of same question are present. 
# split_ratio: Training-Validation Split ratio 
# shuffle_at_end: Shuffle rows after data creation. Useful for select_sample where strategies can be stacked one after the other  
# Prepare Training and Validation Data 
# Inputs: 
# col_name: Dataframe column containing LLM responses. 
# path: data path 
# multiple_strategies: use multiple strategies during training - True only for select_sample  
# arrange_train: arrange training data such that all questions are seen first during training. 
#                Only applicable when multiple generations of same question are present. 
# remove_train_duplicates: remove duplicate questions in training data. Only applicable when multiple generations of same question are present. 
# remove_val_duplicates: remove duplicate questions in validation data. Only applicable when multiple generations of same question are present. 
# split_ratio: Training-Validation Split ratio 
# shuffle_at_end: Shuffle rows after data creation. Useful for select_sample where strategies can be stacked one after the other  
def prepare_data(col_name, path, multiple_strategies=False, arrange_train=False, 
                 remove_train_duplicates=True, remove_val_duplicates=True, split_ratio=0.8,
                 shuffle_at_end=False, dataset_name='gsm8k', eos_token='<|endoftext|>'):
    df = pd.read_csv(path)
    df['question'] = df['question'].apply(str).apply(lambda x: x.strip())
    print("Input Shape: ", df.shape)
    
    # Add if 'id' is not present in input dataframe  
    if 'id' not in df.columns:
        prev_rows = df.shape[0]
        if 'gsm8k' in dataset_name.lower(): 
            ds_hf = datasets.load_dataset(dataset_name,'main',split='train')
        else:  # metamath 
            ds_hf = datasets.load_dataset(dataset_name, split="train")
        df_hf = ds_hf.to_pandas().reset_index(drop=False)
        # Metamath has 'query' column 
        if 'question' not in df_hf.columns:
            df_hf.rename(columns={'query':'question'},inplace=True)
        # Drop duplicated questions
        df_hf = df_hf.drop_duplicates(subset=['question'])
        # Remove whitespaces at end and start 
        df_hf['question'] = df_hf['question'].apply(str).apply(lambda x: x.strip())
        # Add Id column
        df_hf = df_hf.reset_index(drop=False)
        df_hf.rename(columns={'index':'id'}, inplace=True)
        # Merge 
        df = df.merge(df_hf[['id','question']], on = ['question'], how = 'inner')
        # assert(prev_rows == df.shape[0])
        print("Shape after merging: \n", df.shape)
        assert(df.isnull().sum().sum()==0)
    
    if multiple_strategies:
        df['response'] = df.apply(lambda row: '[' + row['strategy'] + ']: ' + row[col_name], axis=1)
    else:
        # Rename 'col_name' to response
        df.rename(columns={col_name:'response'}, inplace=True)
    
    # Get Train and Val DFs
    df_train, df_val = split_train_val(df, split_ratio)
    df_train = df_train.sort_values(by=['id'])
    df_val = df_val.sort_values(by=['id'])
    
    if remove_train_duplicates:
        print("Removing Train Duplicates")
        df_train = df_train.drop_duplicates(subset=['id'])
        df_train.reset_index(drop=True, inplace=True) 
        print(df_train.shape)
        print(df_train.head())
    
    # Arrange DF Train
    if arrange_train:
        df_train = arrange_df(df_train)
    
    # Remove Duplicates for Validation DF 
    if remove_val_duplicates:
        print("Removing Val Duplicates")
        df_val = df_val.drop_duplicates(subset=['id'])
        df_val.reset_index(drop=True, inplace=True)
        print(df_val.shape)
        print(df_val.head())
    
    # Shuffle 
    if shuffle_at_end:
        df_train = df_train.sample(frac=1, random_state=SEED)
        df_val = df_val.sample(frac=1, random_state=SEED)
    
    # Convert to Dataset 
    dataset_train = datasets.Dataset.from_pandas(df_train[['id','question','response']].copy())
    dataset_val = datasets.Dataset.from_pandas(df_val[['id','question','response']].copy())
    
    # Dataset Dict
    ds = datasets.DatasetDict({"train":dataset_train, "val":dataset_val})
    
    print(ds)
    return ds

# Get instruction prompt 
def formatting_prompts_func(examples, prompt, eos_token):
    inputs       = examples["question"]
    outputs      = examples["response"]
    texts = []
    for input, output in zip(inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = prompt.format(input, output) + eos_token
        texts.append(text)
    return { "text" : texts, }


# Function to load model and tokenizer 
def load_model(config):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = config['MODEL']['MODEL_NAME'],
        max_seq_length = config['MODEL']['MAX_SEQ_LENGTH'],
        dtype = config['MODEL']['DTYPE'],
        load_in_4bit = config['MODEL']['LOAD_IN_4BIT']
    )
    return model, tokenizer 

def sample_input_text_for_inference(strategy):
    question = 'Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read?'
    if strategy in ['cot','pot','l2m']:
        input_text = alpaca_prompt.format(question,"")
    else:
        input_text = alpaca_prompt_ss.format(question, "")
    return input_text 


def run(config_path):
    # global variables 
    global SEED 
    
    # Load config 
    config = load_config(config_path)
    
    # Get strategy 
    strategy = config['GLOBAL']['STRATEGY']
    
    # Get seed 
    SEED = config['GLOBAL']['SEED']
    
    # Load model and tokenizer 
    model, tokenizer = load_model(config)
    EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
    print("EOS Token: ", EOS_TOKEN)

    # Prepare data
    colname = config['GLOBAL']['COLNAME']
    data_path = config['GLOBAL']['DATA_PATH']
    dataset_name = config['GLOBAL']['DATASET_HF']
    # Currently no duplicates in the training data, so it doesn't matter if using remove_duplicates for train and val. 
    if strategy in ['cot','pot','l2m','standard']: 
        ds = prepare_data(col_name=colname, path=data_path, 
                          multiple_strategies=False, arrange_train=False, 
                          remove_train_duplicates=False, remove_val_duplicates=True, 
                          split_ratio=config['GLOBAL']['SPLIT_RATIO'], 
                         dataset_name = dataset_name)
        dataset_train = ds['train'].map(lambda x: formatting_prompts_func(x, alpaca_prompt, EOS_TOKEN), batched = True)
        dataset_val = ds['val'].map(lambda x: formatting_prompts_func(x, alpaca_prompt, EOS_TOKEN), batched = True)
    
    # strategy = 'select_sample'
    else: 
        ds = prepare_data(col_name=colname, path=data_path,
                          multiple_strategies=True, arrange_train=False, 
                          remove_train_duplicates=False, remove_val_duplicates=True, 
                          split_ratio=config['GLOBAL']['SPLIT_RATIO'], shuffle_at_end=True,
                         dataset_name=dataset_name)

        dataset_train = ds['train'].map(lambda x: formatting_prompts_func(x, alpaca_prompt_ss, EOS_TOKEN), batched = True)
        dataset_val = ds['val'].map(lambda x: formatting_prompts_func(x, alpaca_prompt_ss, EOS_TOKEN), batched = True)
    
    
    # Save Dataset
    # Saving Train-Val Dataset 
    save_path = os.path.join(config['GLOBAL']['OUTPUT_PATH'], 'ds-train-val-{strat}'.format(strat=strategy))
    ds.save_to_disk(save_path)
    
    print("Dataset Train: \n", dataset_train)
    print("Dataset Val: \n", dataset_val) 
    
    # Check dataset examples 
    print("\nFirst 5 examples in train dataset...\n")
    for i in range(5):
        print(dataset_train['text'][i])
    print("\nFirst 5 examples in validation dataset...\n")
    for i in range(5):
        print(dataset_val['text'][i])

    # LoRA arguments 
    model = FastLanguageModel.get_peft_model(
        model, 
        r = config['LORA']['r'], 
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
        lora_alpha = config['LORA']['lora_alpha'],
        lora_dropout = config['LORA']['lora_dropout'],
        bias = config['LORA']['bias'], 
        use_gradient_checkpointing = config['LORA']['use_gradient_checkpointing'], 
        random_state = SEED, 
        use_rslora = config['LORA']['use_rslora'],
        loftq_config = config['LORA']['loftq_config']
    )

    # Wandb arguments 
    if config['WANDB']['REPORT']:
        WANDB_API_KEY = config['WANDB']['API_KEY'] if config['WANDB']['API_KEY'] else os.environ.get('WANDB_API_KEY', None)
        wandb.login(key=WANDB_API_KEY)
        wandb.init(project=config['WANDB']['PROJECT_NAME'], 
                   name=config['WANDB']['RUN_NAME'], 
                   reinit=True)
    
    # Training arguments 
    training_args_dict = config.get('TRAINING_ARGS', {})
    # Add extra arguments : FP16, BF16, seed, and report_to 
    training_args_dict['fp16'] = not torch.cuda.is_bf16_supported()
    training_args_dict['bf16'] = torch.cuda.is_bf16_supported()
    training_args_dict['seed'] = SEED 
    training_args_dict['report_to'] = "wandb" if config['WANDB']['REPORT'] else None 
    # Set Evaluation Strategy to "no" if DO_EVAL is False
    if not config['GLOBAL']['DO_EVAL']:
        training_args_dict['evaluation_strategy'] = "no"
        training_args_dict["do_eval"] = False 

    print(training_args_dict['report_to'])
    
    # Get training arguments 
    parser = HfArgumentParser(TrainingArguments)
    train_args = parser.parse_dict(training_args_dict)[0]

    # Set up trainer
    if config['GLOBAL']['DO_EVAL']: 
        trainer = SFTTrainer(
            model = model,
            tokenizer = tokenizer,
            train_dataset = dataset_train,
            eval_dataset = dataset_val,
            dataset_text_field = "text",
            max_seq_length = config['MODEL']['MAX_SEQ_LENGTH'],
            # dataset_num_proc = 2,
            packing = True, 
            args = train_args
        )
    else:
        trainer = SFTTrainer(
            model = model,
            tokenizer = tokenizer,
            train_dataset = dataset_train,
            eval_dataset = None,
            dataset_text_field = "text",
            max_seq_length = config['MODEL']['MAX_SEQ_LENGTH'],
            # dataset_num_proc = 2,
            packing = True, 
            args = train_args
        )

    # Train 
    trainer_stats = trainer.train()
    if config['WANDB']['REPORT']:
        wandb.finish()

    # Check inference 
    sample_input_text = sample_input_text_for_inference(strategy)
    sample_inputs = tokenizer(sample_input_text, return_tensors = "pt").to("cuda")
    try:
        sample_outputs = model.generate(**sample_inputs, max_new_tokens = 1024, use_cache = True)
    except:
        print("HF model.generate didn't work. Using Unsloth's FastLanguageModel.for_inference()")
        FastLanguageModel.for_inference(model) 
        sample_outputs = model.generate(**sample_inputs, max_new_tokens = 1024, use_cache = True)
    decoded_sample_outputs = tokenizer.batch_decode(sample_outputs)
    print("Sample Input Text: \n", sample_input_text)
    print("Sample Output Text: \n", decoded_sample_outputs)

    # Save model 
    model.save_pretrained(os.path.join(config['GLOBAL']['OUTPUT_PATH'], "final_model")) # Local saving
    tokenizer.save_pretrained(os.path.join(config['GLOBAL']['OUTPUT_PATH'], "final_model"))
    print("Model Saved")

    # Save merged model 
    #model.save_pretrained_merged(os.path.join(config['GLOBAL']['OUTPUT_PATH'], "final_model_merged_train"))
    #print("Model merged and saved")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unsloth LoRA SFT")
    parser.add_argument("config_path", type=str, help="Path to the configuration file")
    args = parser.parse_args()
    run(args.config_path)