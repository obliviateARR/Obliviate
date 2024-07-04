#!/usr/bin/env python
# coding: utf-8

import sys
import os
import argparse
from pprint import pprint

parser = argparse.ArgumentParser(description="Parser for model configuration")

# parser.add_argument('--adapter_lib_path', type=str, default="./", help='adapter library path')
parser.add_argument('--model_dir', type=str, required=True, help='model directory')
parser.add_argument('--output_dir', type=str, default="./output", help='data directory')
parser.add_argument('--cuda', type=int, default=0, help='cuda device id')
parser.add_argument('--task', type=str, required=True, help='task name')
parser.add_argument('--model_name', type=str, required=True, help='model name')
parser.add_argument('--attack', type=str, required=True, help='backdoor attack')
parser.add_argument('--peft', type=str, required=True, help='peft')
parser.add_argument('--defense', action='store_true', help='apply defense or not')
parser.add_argument('--random_seed', type=int, default=0, help='random seed')
parser.add_argument('--triggers', type=list, default=["cf", "mn", "tq", "qt", "mm", "pt"], help='trigger list')
parser.add_argument('--insert_count', type=int, default=1, help='trigger insert count')
parser.add_argument('--train_sample', type=int, default=6000, help='train sample size')
parser.add_argument('--eval_sample', type=int, default=2000, help='evaluation sample size')
parser.add_argument('--train_batch', type=int, default=16, help='train batch size')
parser.add_argument('--eval_batch', type=int, default=128, help='eval batch size')
parser.add_argument('--max_length', type=int, default=128, help='maximum sequence length')
parser.add_argument('--warmup', type=float, default=0, help='warmup_ratio')
parser.add_argument('--lr_scheduler', type=str, default='linear', help='lr scheduler type')
parser.add_argument('--lr', type=float, required=True, help='learning rate')
parser.add_argument('--epoch', type=int, required=True, help='epoch')
parser.add_argument('--amp', type=float, default=None, help='defense alpha for neuron amplification')
parser.add_argument('--reg', type=float, default=None, help='defense alpha for attention regularization')


args = parser.parse_args()

# sys.path.insert(0, args.adapter_lib_path)
os.environ["CUDA_VISIBLE_DEVICES"] = f'{args.cuda}'

import json
import random
import numpy as np

import torch
import torch.nn as nn
# import torch.nn.functional as F

from transformers import AutoTokenizer, TrainingArguments, default_data_collator, set_seed
from transformers.adapters import AutoAdapterModel

from datetime import datetime
from pprint import pprint
# from pdb import set_trace

from utils.data_utils import *
from utils.poison_utils import *
from trainer import *

from utils.create_config import get_config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device_count = torch.cuda.device_count()

current_time = datetime.now().strftime('%Y%m%d-%H%M%S')

task_name = args.task
model_name = args.model_name
attack = args.attack
peft = args.peft
defense = args.defense

attacker_name = f'{attack}_{task_name}'
max_seq_length = args.max_length

suffix = 'eval_defense' if defense else 'eval'
output_dir = os.path.join(args.output_dir, f'{model_name}/{attack}_{peft}_{suffix}/{model_name}_{attacker_name}')


# without defense
if defense:
    assert(args.amp)
    assert(args.reg)
else:
    assert(args.amp == None)
    assert(args.reg == None)


# sample config
train_sample_size = args.train_sample
eval_sample_size = args.eval_sample

# attack config
model_name_or_path = os.path.join(args.model_dir, f'{model_name}_{attack}')
triggers = args.triggers
times = args.insert_count

# defense config
defense_alpha_amp = args.amp
defense_alpha_attn = args.reg

peft_config = get_config(f'{model_name}_{peft}', defense)


# training configuration
num_labels = get_num_labels(task_name)
random_seed = args.random_seed
per_device_train_batch_size = args.train_batch
per_device_eval_batch_size = args.eval_batch
learning_rate = args.lr
num_train_epochs = args.epoch
lr_scheduler_type = args.lr_scheduler
warmup_ratio = args.warmup

set_seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

print(f'[Output Dir] {output_dir}')
print(f'Defense: {defense}')
pprint(peft_config, sort_dicts=False)

if __name__ == '__main__':
    ## data processing
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
    )
    
    raw_datasets = load_dataset_with_glue(task_name)
    
    poison_sentence_key = get_poison_key(task_name)
        
    raw_datasets = get_LMSanitator_split(raw_datasets, task_name)
    
    _train_dataset_clean = get_sample(raw_datasets['train'], sample_size=train_sample_size)
    _eval_dataset_clean = get_sample(get_eval_dataset(raw_datasets, task_name), sample_size=eval_sample_size)
    
    _train_dataset_clean = add_idx(_train_dataset_clean)
    _eval_dataset_clean = add_idx(_eval_dataset_clean)
    
    _train_dataset_clean = align_label(_train_dataset_clean, task_name)
    _eval_dataset_clean = align_label(_eval_dataset_clean, task_name)
        
    _train_dataset_poison = poison_data(_train_dataset_clean, triggers, p=0, times=times, dup_clean=False, sentence_key=poison_sentence_key)[0]
    _eval_dataset_poison = poison_data(_eval_dataset_clean, triggers, p=1, times=times, dup_clean=True, sentence_key=poison_sentence_key)[0]
        
    train_dataset_poison = get_data(_train_dataset_poison, task_name, max_seq_length, tokenizer)
    eval_dataset_poison = get_data(_eval_dataset_poison, task_name, max_seq_length, tokenizer)
    
    train_dataset_poison = train_dataset_poison.map(add_trigger_label, fn_kwargs={'target_words': triggers, 'tokenizer': tokenizer})
    eval_dataset_poison = eval_dataset_poison.map(add_trigger_label, fn_kwargs={'target_words': triggers, 'tokenizer': tokenizer})
    
    print(raw_datasets)
    
    print(train_dataset_poison)
    for l in range(num_labels):
        print(f'Label {l}:', train_dataset_poison['label'].count(l))
    print('Poisoned:', train_dataset_poison['poisoned'].count(1))
    
    print(eval_dataset_poison)
    for l in range(num_labels):
        print(f'Label {l}:', eval_dataset_poison['label'].count(l))
    print('Poisoned:', eval_dataset_poison['poisoned'].count(1))
    
    
    # load PEFT layers
    model = AutoAdapterModel.from_pretrained(model_name_or_path)
    print(f'Load model: {model_name_or_path}')
    model.add_adapter(attacker_name, peft_config)
    if peft == 'lora':
        model.merge_adapter(attacker_name)
        model.reset_adapter()
    model.train_adapter([attacker_name])
    model.add_classification_head(attacker_name, num_labels=num_labels)
    model.active_head = attacker_name
    
    print(model.adapter_summary())
    print(model.active_head)
    
    total_params = format(sum(p.numel() for p in model.parameters()), ',')
    total_params_train = format(sum(p.numel() for p in model.parameters() if p.requires_grad), ',')
    print(f'{total_params_train} / {total_params}')
    
    
    # load trainer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    total_batch_size_train = per_device_train_batch_size * device_count
    total_batch_size_eval = per_device_eval_batch_size * device_count
    
    training_args = TrainingArguments(
        remove_unused_columns=False,
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        num_train_epochs=num_train_epochs,
        logging_dir=None,
        seed=random_seed,
        data_seed=random_seed,
        do_train=True,
        do_eval=True,
        learning_rate=learning_rate,
        lr_scheduler_type=lr_scheduler_type,
        warmup_ratio=warmup_ratio,
        evaluation_strategy='epoch',
        logging_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=1,
        metric_for_best_model = 'loss'
    )
    
    trainer = DefenseTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset_poison,
            eval_dataset=eval_dataset_poison,
            tokenizer=tokenizer,
            data_collator=default_data_collator,
            compute_metrics=None,
            num_labels=num_labels,
            target_words=triggers,
            defense_alpha_amp=defense_alpha_amp,
            defense_alpha_attn=defense_alpha_attn,
            peft=peft,
            scale_calibrate_ratio=((len(eval_dataset_poison)/(len(triggers)+1))//total_batch_size_eval),
        )
    
    
    # training
    os.makedirs(output_dir, exist_ok=True)

    config = vars(args)
    config_add = {'max_seq_length': max_seq_length,
                  'total_batch_size': total_batch_size_train}
    
    config.update(config_add)
    
    with open(os.path.join(output_dir, "hyperparameters.json"), "w") as f:
        json.dump(config, f, indent=4)
    
    train_result = trainer.train()
    metrics = train_result.metrics
    
    trainer.save_model()
    
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    
    os.makedirs(os.path.join(output_dir, f"trained_adapter"), exist_ok=True)
    model.save_adapter(os.path.join(output_dir, f"trained_adapter/{attacker_name}"), attacker_name)
    
    os.makedirs(os.path.join(output_dir, f"trained_head"), exist_ok=True)
    model.save_head(os.path.join(output_dir, f"trained_head/{attacker_name}"), attacker_name)
    
    
    # evaluation
    if peft == 'prefix':
        model.eject_prefix_tuning(attacker_name)
    metrics = trainer.evaluate(eval_dataset=eval_dataset_poison)
    
    print(f'Dataset: {task_name}')
    pprint(metrics)
    
    trainer.save_metrics('eval', metrics)