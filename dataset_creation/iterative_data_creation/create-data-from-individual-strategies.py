"""
Code to create data from individual strategies for epoch 2 
1. Load data from the three strategies 
2. Check which reasoning chains are correct. 
3. Subset correct reasoning chains. One for each 
"""

import pandas as pd 
import datasets 
import json 
import os 
import numpy as np 
import signal 
import re 
import ast 
import string
translator = str.maketrans('', '', string.punctuation)

# Qwen 
# BASE_DIR = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/qwen/'
# COT_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/qwen/gsm8k_train_cot.jsonl'
# POT_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/qwen/gsm8k_train_pot.jsonl'
# L2M_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/qwen/gsm8k_train_l2m.jsonl'

# Gemma-2b-epoch3
# BASE_DIR = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/gemma-epoch3/'
# COT_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/gemma-epoch3/df_forced_preds_cot.csv'
# POT_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/gemma-epoch3/df_forced_preds_pot.csv'
# L2M_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/gemma-epoch3/df_forced_preds_l2m.csv'

# Gemma-7B-epoch2
# BASE_DIR = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/gemma-7b-It/epoch2'
# COT_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/gemma-7b-It/epoch2/df_forced_preds_cot_train.csv'
# POT_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/gemma-7b-It/epoch2/df_forced_preds_pot_train.csv'
# L2M_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/gemma-7b-It/epoch2/df_forced_preds_l2m_train.csv'

# Gemma-7B-epoch3
# BASE_DIR = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/gemma-7b-epoch3'
# COT_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/gemma-7b-epoch3/df_forced_preds_cot_train.csv'
# POT_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/gemma-7b-epoch3/df_forced_preds_pot_train.csv'
# L2M_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/gemma-7b-epoch3/df_forced_preds_l2m_train.csv'

# Qwen2-1.5b-epoch2 
# BASE_DIR = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/qwen2-1.5b-epoch2'
# COT_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/qwen2-1.5b-epoch2/df_forced_preds_cot_train.csv'
# POT_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/qwen2-1.5b-epoch2/df_forced_preds_pot_train.csv'
# L2M_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/qwen2-1.5b-epoch2/df_forced_preds_l2m_train.csv'

# Qwen2-0.5b-epoch2
# BASE_DIR = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/qwen2-0.5b-epoch2'
# COT_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/qwen2-0.5b-epoch2/df_forced_preds_cot_train.csv'
# POT_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/qwen2-0.5b-epoch2/df_forced_preds_pot_train.csv'
# L2M_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/qwen2-0.5b-epoch2/df_forced_preds_l2m_train.csv'

# Qwen2-0.5b-epoch3
# BASE_DIR = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/qwen2-0.5b-epoch3'
# COT_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/qwen2-0.5b-epoch3/df_forced_preds_cot_train.csv'
# POT_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/qwen2-0.5b-epoch3/df_forced_preds_pot_train.csv'
# L2M_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/qwen2-0.5b-epoch3/df_forced_preds_l2m_train.csv'

# Qwen2-1.5b-epoch3
# BASE_DIR = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/qwen2-1.5b-epoch3'
# COT_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/qwen2-1.5b-epoch3/df_forced_preds_cot_train.csv'
# POT_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/qwen2-1.5b-epoch3/df_forced_preds_pot_train.csv'
# L2M_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/qwen2-1.5b-epoch3/df_forced_preds_l2m_train.csv'

# Llama3-70b-Qwen2-0.5B-iteration2
# BASE_DIR = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/qwen2-0.5b-iteration2/selsamp-biased'
# COT_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/qwen2-0.5b-iteration2/selsamp-biased/df_forced_preds_cot_train.csv'
# POT_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/qwen2-0.5b-iteration2/selsamp-biased/df_forced_preds_pot_train.csv'
# L2M_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/qwen2-0.5b-iteration2/selsamp-biased/df_forced_preds_l2m_train.csv'

# Llama3-70b-Qwen2-1.5B-iteration2
# BASE_DIR = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/qwen2-1.5b-iteration2/selsamp-biased'
# COT_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/qwen2-1.5b-iteration2/selsamp-biased/df_forced_preds_cot_train.csv'
# POT_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/qwen2-1.5b-iteration2/selsamp-biased/df_forced_preds_pot_train.csv'
# L2M_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/qwen2-1.5b-iteration2/selsamp-biased/df_forced_preds_l2m_train.csv'

# Llama3-70b-Gemma-2b-iteration2
# BASE_DIR = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/gemma-2b-iteration2/selsamp-biased'
# COT_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/gemma-2b-iteration2/selsamp-biased/df_forced_preds_cot_train.csv'
# POT_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/gemma-2b-iteration2/selsamp-biased/df_forced_preds_pot_train.csv'
# L2M_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/gemma-2b-iteration2/selsamp-biased/df_forced_preds_l2m_train.csv'

# Llama3-70b-Gemma-7b-iteration2
# BASE_DIR = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/gemma-7b-iteration2/selsamp-biased'
# COT_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/gemma-7b-iteration2/selsamp-biased/df_forced_preds_cot_train.csv'
# POT_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/gemma-7b-iteration2/selsamp-biased/df_forced_preds_pot_train.csv'
# L2M_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/gemma-7b-iteration2/selsamp-biased/df_forced_preds_l2m_train.csv'

# Llama3-70b-Qwen2-0.5b-iteration3-standard 
# BASE_DIR = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/qwen2-0.5b-iteration3/standard'
# COT_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/qwen2-0.5b-iteration3/standard/df_forced_preds_cot_train.csv'
# POT_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/qwen2-0.5b-iteration3/standard/df_forced_preds_pot_train.csv'
# L2M_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/qwen2-0.5b-iteration3/standard/df_forced_preds_l2m_train.csv'

# Llama3-70b-Qwen2-0.5b-iteration3-mixed 
# BASE_DIR = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/qwen2-0.5b-iteration3/mixed-data'
# COT_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/qwen2-0.5b-iteration3/mixed-data/df_forced_preds_cot_train.csv'
# POT_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/qwen2-0.5b-iteration3/mixed-data/df_forced_preds_pot_train.csv'
# L2M_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/qwen2-0.5b-iteration3/mixed-data/df_forced_preds_l2m_train.csv'

# # Llama3-70b-Qwen2-1.5b-iteration3-standard 
# BASE_DIR = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/qwen2-1.5b-iteration3/standard'
# COT_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/qwen2-1.5b-iteration3/standard/df_forced_preds_cot_train.csv'
# POT_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/qwen2-1.5b-iteration3/standard/df_forced_preds_pot_train.csv'
# L2M_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/qwen2-1.5b-iteration3/standard/df_forced_preds_l2m_train.csv'

# Llama3-70b-Qwen2-1.5b-iteration3-mixed
# BASE_DIR = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/qwen2-1.5b-iteration3/mixed-data'
# COT_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/qwen2-1.5b-iteration3/mixed-data/df_forced_preds_cot_train.csv'
# POT_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/qwen2-1.5b-iteration3/mixed-data/df_forced_preds_pot_train.csv'
# L2M_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/qwen2-1.5b-iteration3/mixed-data/df_forced_preds_l2m_train.csv'

# Llama3-70b-Gemma2b-iteration3-standard
# BASE_DIR = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/gemma-2b-iteration3/standard'
# COT_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/gemma-2b-iteration3/standard/df_forced_preds_cot_train.csv'
# POT_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/gemma-2b-iteration3/standard/df_forced_preds_pot_train.csv'
# L2M_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/gemma-2b-iteration3/standard/df_forced_preds_l2m_train.csv'

# Llama3-70b-Gemma2b-iteration3-mixed 
# BASE_DIR = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/gemma-2b-iteration3/mixed-data'
# COT_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/gemma-2b-iteration3/mixed-data/df_forced_preds_cot_train.csv'
# POT_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/gemma-2b-iteration3/mixed-data/df_forced_preds_pot_train.csv'
# L2M_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/gemma-2b-iteration3/mixed-data/df_forced_preds_l2m_train.csv'

# Llama3-70b-Gemma7b-iteration3-standard
# BASE_DIR = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/gemma-7b-iteration3/standard'
# COT_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/gemma-7b-iteration3/standard/df_forced_preds_cot_train.csv'
# POT_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/gemma-7b-iteration3/standard/df_forced_preds_pot_train.csv'
# L2M_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/gemma-7b-iteration3/standard/df_forced_preds_l2m_train.csv'

# Llama3-70b-Gemma7b-iteration3-mixed
# BASE_DIR = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/gemma-7b-iteration3/mixed-data'
# COT_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/gemma-7b-iteration3/mixed-data/df_forced_preds_cot_train.csv'
# POT_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/gemma-7b-iteration3/mixed-data/df_forced_preds_pot_train.csv'
# L2M_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/gemma-7b-iteration3/mixed-data/df_forced_preds_l2m_train.csv'

# Llama3-70b-Qwen2-0.5b-iteration2 (using selsamp generated data) 
# BASE_DIR = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/qwen2-0.5b-iteration2/selsamp'
# COT_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/qwen2-0.5b-iteration2/selsamp/df_forced_preds_cot_train.csv'
# POT_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/qwen2-0.5b-iteration2/selsamp/df_forced_preds_pot_train.csv'
# L2M_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/qwen2-0.5b-iteration2/selsamp/df_forced_preds_l2m_train.csv'

# Llama3-70b-Qwen2-1.5b-iteration2 (using selsamp generated data) 
# BASE_DIR = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/qwen2-1.5b-iteration2/selsamp'
# COT_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/qwen2-1.5b-iteration2/selsamp/df_forced_preds_cot_train.csv'
# POT_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/qwen2-1.5b-iteration2/selsamp/df_forced_preds_pot_train.csv'
# L2M_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/qwen2-1.5b-iteration2/selsamp/df_forced_preds_l2m_train.csv'

# Llama3-70b-Gemma2-7b-iteration2 (using selsamp generated data) 
# BASE_DIR = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/gemma-7b-iteration2/selsamp'
# COT_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/gemma-7b-iteration2/selsamp/df_forced_preds_cot_train.csv'
# POT_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/gemma-7b-iteration2/selsamp/df_forced_preds_pot_train.csv'
# L2M_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/gemma-7b-iteration2/selsamp/df_forced_preds_l2m_train.csv'

# Llama3-70b-Gemma2-7b-iteration2 (using selsamp generated data) 
# BASE_DIR = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/gemma-2b-iteration2/selsamp'
# COT_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/gemma-2b-iteration2/selsamp/df_forced_preds_cot_train.csv'
# POT_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/gemma-2b-iteration2/selsamp/df_forced_preds_pot_train.csv'
# L2M_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/gemma-2b-iteration2/selsamp/df_forced_preds_l2m_train.csv'
# EPOCH = 2

# Llama3-70b-Gemma2b-iteration4-standard
# BASE_DIR = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/gemma-2b-iteration4/standard'
# COT_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/gemma-2b-iteration4/standard/df_forced_preds_cot_train.csv'
# POT_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/gemma-2b-iteration4/standard/df_forced_preds_pot_train.csv'
# L2M_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/gemma-2b-iteration4/standard/df_forced_preds_l2m_train.csv'

# Llama3-70b-Gemma2b-iteration4-mixed
# BASE_DIR = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/gemma-2b-iteration4/mixed'
# COT_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/gemma-2b-iteration4/mixed/df_forced_preds_cot_train.csv'
# POT_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/gemma-2b-iteration4/mixed/df_forced_preds_pot_train.csv'
# L2M_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/gemma-2b-iteration4/mixed/df_forced_preds_l2m_train.csv'

# Llama3-70b-Gemma7b-iteration4-mixed
# BASE_DIR = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/gemma-7b-iteration4/mixed-data'
# COT_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/gemma-7b-iteration4/mixed-data/df_forced_preds_cot_train.csv'
# POT_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/gemma-7b-iteration4/mixed-data/df_forced_preds_pot_train.csv'
# L2M_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/gemma-7b-iteration4/mixed-data/df_forced_preds_l2m_train.csv'

# Llama3-70b-Gemma7b-iteration4-standard
# BASE_DIR = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/gemma-7b-iteration4/standard'
# COT_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/gemma-7b-iteration4/standard/df_forced_preds_cot_train.csv'
# POT_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/gemma-7b-iteration4/standard/df_forced_preds_pot_train.csv'
# L2M_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/gemma-7b-iteration4/standard/df_forced_preds_l2m_train.csv'

# Llama3-70b-Qwen2-0.5B-iteration4-standard
# BASE_DIR = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/qwen2-0.5b-iteration4/standard'
# COT_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/qwen2-0.5b-iteration4/standard/df_forced_preds_cot_train.csv'
# POT_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/qwen2-0.5b-iteration4/standard/df_forced_preds_pot_train.csv'
# L2M_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/qwen2-0.5b-iteration4/standard/df_forced_preds_l2m_train.csv'

# Llama3-70b-Qwen2-0.5B-iteration4-mixed-data
# BASE_DIR = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/qwen2-0.5b-iteration4/mixed-data'
# COT_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/qwen2-0.5b-iteration4/mixed-data/df_forced_preds_cot_train.csv'
# POT_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/qwen2-0.5b-iteration4/mixed-data/df_forced_preds_pot_train.csv'
# L2M_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/qwen2-0.5b-iteration4/mixed-data/df_forced_preds_l2m_train.csv'

# Llama3-70b-Qwen2-1.5B-iteration4-standard
# BASE_DIR = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/qwen2-1.5b-iteration4/standard'
# COT_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/qwen2-1.5b-iteration4/standard/df_forced_preds_cot_train.csv'
# POT_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/qwen2-1.5b-iteration4/standard/df_forced_preds_pot_train.csv'
# L2M_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/qwen2-1.5b-iteration4/standard/df_forced_preds_l2m_train.csv'

# Llama3-70b-Qwen2-1.5B-iteration4-mixed-data
# BASE_DIR = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/qwen2-1.5b-iteration4/mixed-data'
# COT_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/qwen2-1.5b-iteration4/mixed-data/df_forced_preds_cot_train.csv'
# POT_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/qwen2-1.5b-iteration4/mixed-data/df_forced_preds_pot_train.csv'
# L2M_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/qwen2-1.5b-iteration4/mixed-data/df_forced_preds_l2m_train.csv'

# Llama3-70b-SmolLM-1.7B-iteration2 (selsamp-biased)
# BASE_DIR = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/smollm-1.7b-iteration2/selsamp-biased'
# COT_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/smollm-1.7b-iteration2/selsamp-biased/df_forced_preds_cot_train.csv'
# POT_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/smollm-1.7b-iteration2/selsamp-biased/df_forced_preds_pot_train.csv'
# L2M_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/smollm-1.7b-iteration2/selsamp-biased/df_forced_preds_l2m_train.csv'

# Llama3-70b-SmolLM-1.7B-iteration3 (standard)
# BASE_DIR = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/smollm-1.7b-iteration3/standard'
# COT_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/smollm-1.7b-iteration3/standard/df_forced_preds_cot_train.csv'
# POT_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/smollm-1.7b-iteration3/standard/df_forced_preds_pot_train.csv'
# L2M_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/smollm-1.7b-iteration3/standard/df_forced_preds_l2m_train.csv'

# Llama3-70b-SmolLM-1.7B-iteration3 (mixed-data)
# BASE_DIR = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/smollm-1.7b-iteration3/mixed-data'
# COT_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/smollm-1.7b-iteration3/mixed-data/df_forced_preds_cot_train.csv'
# POT_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/smollm-1.7b-iteration3/mixed-data/df_forced_preds_pot_train.csv'
# L2M_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/smollm-1.7b-iteration3/mixed-data/df_forced_preds_l2m_train.csv'

# Llama3-70b-SmolLM-1.7B-iteration4 (standard)
# BASE_DIR = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/smollm-1.7b-iteration4/standard'
# COT_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/smollm-1.7b-iteration4/standard/df_forced_preds_cot_train.csv'
# POT_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/smollm-1.7b-iteration4/standard/df_forced_preds_pot_train.csv'
# L2M_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/smollm-1.7b-iteration4/standard/df_forced_preds_l2m_train.csv'

# Llama3-70b-SmolLM-1.7B-iteration4 (mixed)
# BASE_DIR = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/smollm-1.7b-iteration4/mixed-data'
# COT_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/smollm-1.7b-iteration4/mixed-data/df_forced_preds_cot_train.csv'
# POT_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/smollm-1.7b-iteration4/mixed-data/df_forced_preds_pot_train.csv'
# L2M_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/smollm-1.7b-iteration4/mixed-data/df_forced_preds_l2m_train.csv'
# EPOCH = 4

# Llama3-70b-Gemma2b-iteration3 (all)
# BASE_DIR = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/gemma-2b-iteration3/all'
# COT_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/gemma-2b-iteration3/all/df_forced_preds_cot_train.csv'
# POT_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/gemma-2b-iteration3/all/df_forced_preds_pot_train.csv'
# L2M_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/gemma-2b-iteration3/all/df_forced_preds_l2m_train.csv'
# EPOCH = 3

# # Llama3-70b-Qwen2-0.5b-iteration3 (all)
# BASE_DIR = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/qwen2-0.5b-iteration3/all'
# COT_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/qwen2-0.5b-iteration3/all/df_forced_preds_cot_train.csv'
# POT_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/qwen2-0.5b-iteration3/all/df_forced_preds_pot_train.csv'
# L2M_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/qwen2-0.5b-iteration3/all/df_forced_preds_l2m_train.csv'
# EPOCH = 3

# # Llama3-70b-Qwen2-1.5b-iteration3 (all)
# BASE_DIR = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/qwen2-1.5b-iteration3/all'
# COT_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/qwen2-1.5b-iteration3/all/df_forced_preds_cot_train.csv'
# POT_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/qwen2-1.5b-iteration3/all/df_forced_preds_pot_train.csv'
# L2M_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/qwen2-1.5b-iteration3/all/df_forced_preds_l2m_train.csv'
# EPOCH = 3

# # Llama3-70b-SmolLM-1.7b-iteration3 (all)
# BASE_DIR = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/smollm-1.7b-iteration3/all'
# COT_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/smollm-1.7b-iteration3/all/df_forced_preds_cot_train.csv'
# POT_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/smollm-1.7b-iteration3/all/df_forced_preds_pot_train.csv'
# L2M_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/smollm-1.7b-iteration3/all/df_forced_preds_l2m_train.csv'
# EPOCH = 3

# # Llama3-70b-Gemma-2b-iteration5 (mixed)
# BASE_DIR = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/gemma-2b-iteration5/mixed-data'
# COT_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/gemma-2b-iteration5/mixed-data/df_forced_preds_cot_train.csv'
# POT_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/gemma-2b-iteration5/mixed-data/df_forced_preds_pot_train.csv'
# L2M_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/gemma-2b-iteration5/mixed-data/df_forced_preds_l2m_train.csv'
# EPOCH = 5

# #Llama3-70b-Gemma-2b-iteration5 (standard)
# BASE_DIR = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/gemma-2b-iteration5/standard'
# COT_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/gemma-2b-iteration5/standard/df_forced_preds_cot_train.csv'
# POT_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/gemma-2b-iteration5/standard/df_forced_preds_pot_train.csv'
# L2M_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/gemma-2b-iteration5/standard/df_forced_preds_l2m_train.csv'
# EPOCH = 5

# Llama3-70b-Gemma-2b-iteration4 (all)
# BASE_DIR = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/gemma-2b-iteration4/all'
# COT_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/gemma-2b-iteration4/all/df_forced_preds_cot_train.csv'
# POT_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/gemma-2b-iteration4/all/df_forced_preds_pot_train.csv'
# L2M_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/gemma-2b-iteration4/all/df_forced_preds_l2m_train.csv'
# EPOCH = 4

# # Llama3-70b-Qwen2-0.5b-iteration4 (all)
# BASE_DIR = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/qwen2-0.5b-iteration4/all'
# COT_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/qwen2-0.5b-iteration4/all/df_forced_preds_cot_train.csv'
# POT_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/qwen2-0.5b-iteration4/all/df_forced_preds_pot_train.csv'
# L2M_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/qwen2-0.5b-iteration4/all/df_forced_preds_l2m_train.csv'
# EPOCH = 4

# # Llama3-70b-Qwen2-1.5b-iteration4 (all)
# BASE_DIR = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/qwen2-1.5b-iteration4/all'
# COT_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/qwen2-1.5b-iteration4/all/df_forced_preds_cot_train.csv'
# POT_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/qwen2-1.5b-iteration4/all/df_forced_preds_pot_train.csv'
# L2M_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/qwen2-1.5b-iteration4/all/df_forced_preds_l2m_train.csv'
# EPOCH = 4

# # Llama3-70b-SmolLM-1.7b-iteration4 (all)
# BASE_DIR = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/smollm-1.7b-iteration4/all'
# COT_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/smollm-1.7b-iteration4/all/df_forced_preds_cot_train.csv'
# POT_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/smollm-1.7b-iteration4/all/df_forced_preds_pot_train.csv'
# L2M_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/smollm-1.7b-iteration4/all/df_forced_preds_l2m_train.csv'
# EPOCH = 4

# Llama3-70b-Gemma7B-iteration3 (all)
BASE_DIR = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/gemma-7b-iteration3/all'
COT_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/gemma-7b-iteration3/all/df_forced_preds_cot_train.csv'
POT_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/gemma-7b-iteration3/all/df_forced_preds_pot_train.csv'
L2M_PATH = '/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/llama3-70b-any-at-10/gemma-7b-iteration3/all/df_forced_preds_l2m_train.csv'
EPOCH = 3

def read_data(file_path, strategy):
    if file_path.endswith(".jsonl"): 
        data = []
        # Open the file in read mode
        with open(file_path, 'r', encoding='utf-8') as file:
            # Read each line in the file
            for line in file:
                # Parse the line as JSON and add to the list
                data.append(json.loads(line))
        df = pd.DataFrame(data)
    
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    
    if 'id' not in df.columns:
        df = df.reset_index(drop=False)
        df.rename(columns={'index':'id'}, inplace=True)
    df['strategy'] = [strategy] * len(df)
    return df 

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

def expand_df(df, colname='predicted_label'):
    df_expanded = df.explode(colname).reset_index(drop=True)
    # Check if '[cot]', '[pot]' or '[l2m]' is present in predictions 
    prediction0 = df[colname].iloc[0][0]
    if '[cot]' in prediction0 or '[pot]' in prediction0 or '[l2m]' in prediction0:
        df_expanded['strategy'] = df_expanded[colname].apply(lambda x: x.split(':')[0].replace('\n','').replace('[','').replace(']',''))
        df_expanded['predictions'] = df_expanded[colname].apply(lambda x: x.replace('[cot]:','').replace('[pot]:','').replace('[l2m]:','').strip())
    else:
        df_expanded.rename(columns={colname:'predictions'}, inplace=True)
    # import pdb ; pdb.set_trace()
    print("CoT: ", df_expanded['strategy'].tolist().count('cot'))
    print("PoT: ", df_expanded['strategy'].tolist().count('pot'))
    print("L2M: ", df_expanded['strategy'].tolist().count('l2m'))
    return df_expanded


def get_true_num_answer(df, colname='response'):
    if 'answer' in df.columns:
        df['answer'] = df['answer'].apply(float)
        return df 
    df['answer'] = df[colname].apply(lambda x: float(x.split("\n")[-1].replace("#",'').replace(",", "").replace(" ", "").replace("\n", "").replace("$", "").replace("x", "").strip()))
    return df 

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
def get_accuracy(df):
    try:
        df['answer'] = df['answer'].apply(float)
    except:
        df['answer'] = pd.to_numeric(df['answer'], errors='coerce')
        print("Removed {x} rows with non-float true answers".format(x=df['answer'].isnull().sum()))
        df = df.dropna()
    
    df['predicted_num_answer'] = df['predictions'].apply(lambda x: get_pred_answers(x))
    not_found = (df['predicted_num_answer']==-2048).sum()
    print("Predicted answer not found for {x} samples".format(x=not_found))
    df['is_correct'] = df.apply(lambda row: float(row['answer'])==float(row['predicted_num_answer']), axis=1)
    acc = (df['is_correct'].sum(), np.round(100*df['is_correct'].sum()/df.shape[0], 2))
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
        if len(ans) > 0:
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
    predictions = df['predictions'].tolist()
    
    # Format predictions 
    formatted_predictions = format_predictions(predictions)
    
    # Get Numeric answers from formatted predictions 
    predicted_answers = [extract_pred_math_solver(formatted_prediction) for formatted_prediction in formatted_predictions]
    df['predicted_num_answer'] = predicted_answers
    not_found = predicted_answers.count(-2048)
    print("Predicted answer not found for {x} samples".format(x=not_found))
    df['is_correct'] = df.apply(lambda row: is_correct_helper(row['answer'], row['predicted_num_answer']), axis=1)
    acc = (df['is_correct'].sum(), np.round(100*df['is_correct'].sum()/df.shape[0], 2))
    return df, acc 

def distill(df):
    print("Shape before distilling: ", df.shape)
    df_distilled = df[df['is_correct']].copy()
    print("Shape after distilling: ", df_distilled.shape)
    return df_distilled

def select_one_rc(df, subset_cols=['id']):
    df = df.drop_duplicates(subset=subset_cols)
    df.reset_index(drop=True, inplace=True)
    return df 

def run(path, prediction_colname):
    # import pdb ; pdb.set_trace()
    # Read
    if 'cot' in path: 
        df = read_data(path, 'cot')
    elif 'pot' in path:
        df = read_data(path, 'pot')
    elif 'l2m' in path:
        df = read_data(path, 'l2m')
    else:
        print("Incorrect Path")
        return 
    nrows = len(df)
    # Typecast predictions to list
    if isinstance(df[prediction_colname].iloc[0], str):
        df[prediction_colname] = df[prediction_colname].apply(lambda x: ast.literal_eval(x))

    # Expand 
    df = expand_df(df, colname=prediction_colname)
    # Remove noise from L2M Reasoning Chains 
    if 'l2m' in path:
        df = select_l2m_rc(df, 'predictions')

    # Get True Answers 
    df = get_true_num_answer(df)
    
    # Get Predicted Answer 
    if 'cot' in path or 'l2m' in path:
        df, acc = get_accuracy(df)
    # PoT 
    else:
        df, acc = get_accuracy_pot(df)
    
    # Distill 
    df = distill(df)
    assert(df['is_correct'].sum() == df.shape[0])
    
    # Select one reasoning chain per question
    df = select_one_rc(df)
    acc = np.round(100 * df.shape[0]/nrows, 2)
    print("Accuracy: ", np.round(100 * df.shape[0]/nrows, 2))
    return df, acc  

def save(df, input_path, epoch, concat=False):
    last_slash_index = input_path.rfind('/')
    dir_path = input_path[:last_slash_index]
    if concat:
        df.to_csv(os.path.join(dir_path, 'df-combined-distilled-epoch{x}.csv'.format(x=epoch)), index=False)
    elif 'cot' in input_path:
        df.to_csv(os.path.join(dir_path, 'df-cot-distilled-epoch{x}.csv'.format(x=epoch)), index=False)
    elif 'l2m' in input_path:
        df.to_csv(os.path.join(dir_path, 'df-l2m-distilled-epoch{x}.csv'.format(x=epoch)), index=False)
    elif 'pot' in input_path:
        df.to_csv(os.path.join(dir_path, 'df-pot-distilled-epoch{x}.csv'.format(x=epoch)), index=False)
    else:
        print("No Save path found")

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

if __name__ == "__main__":
    print("CoT")
    df_cot, acc_cot = run(COT_PATH, 'predictions')
    save(df_cot, input_path=COT_PATH, epoch=EPOCH)
    print(df_cot.head())
    
    print("L2M")
    df_l2m, acc_l2m = run(L2M_PATH, 'predictions')
    save(df_l2m, input_path=L2M_PATH, epoch=EPOCH)
    print(df_l2m.head())
    
    print("PoT")
    df_pot, acc_pot = run(POT_PATH, 'predictions')
    save(df_pot, input_path=POT_PATH, epoch=EPOCH)
    print(df_pot.head())
    print("CoT accuracy: ", acc_cot)
    print("PoT accuracy: ", acc_pot)
    print("L2M accuracy: ", acc_l2m)
    
    print("Concat")
    df_concat = pd.concat([df_cot, df_pot, df_l2m], axis=0).reset_index(drop=True)
    df_concat.rename(columns={'query':'question', 'predictions':'output_answer', 
                              'answer':'correct_answer', 'predicted_num_answer':'llm_numeric_ans'}, inplace=True)
    save(df_concat, COT_PATH, epoch=EPOCH, concat=True)
    print(df_concat.shape)
    print(df_concat.columns)
    print(df_concat.head())
    
    # Get Biased Data 
    # df_concat = pd.read_csv('/Users/shivamadarsh/Documents/Studies/sem-4/master-thesis/improve-prompt-strategies/data/knowledge_distillation/action-reward/annotation/gemma/df-combined-distilled-epoch2.csv')
    
    # PoT Biased 
    df_pot_biased = get_biased_data(df_concat, 'pot')
    print(df_pot_biased.columns)
    print(df_pot_biased.shape)
    print(df_pot_biased['output_answer'].iloc[0])
    print(df_pot_biased.head())
    df_pot_biased.to_csv(os.path.join(BASE_DIR, 'df-pot-biased-distilled-epoch{x}.csv'.format(x=EPOCH)), index=False)

    # CoT Biased 
    df_cot_biased = get_biased_data(df_concat, 'cot')
    print(df_cot_biased.columns)
    print(df_cot_biased.shape)
    print(df_cot_biased['output_answer'].iloc[0])
    print(df_cot_biased.head())
    df_cot_biased.to_csv(os.path.join(BASE_DIR, 'df-cot-biased-distilled-epoch{x}.csv'.format(x=EPOCH)), index=False)

    # CoT Biased 
    df_l2m_biased = get_biased_data(df_concat, 'l2m')
    print(df_l2m_biased.columns)
    print(df_l2m_biased.shape)
    print(df_l2m_biased['output_answer'].iloc[0])
    print(df_l2m_biased.head())
    df_l2m_biased.to_csv(os.path.join(BASE_DIR, 'df-l2m-biased-distilled-epoch{x}.csv'.format(x=EPOCH)), index=False)

