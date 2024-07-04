import torch

import datasets
import numpy as np
import random
from datasets import load_dataset, DatasetDict, ClassLabel

is_glue = {"cola": True,
            "mnli": True,
            "mrpc": True,
            "qnli": True,
             "qqp": True,
             "rte": True,
            "sst2": True,
            "stsb": True,
            "wnli": True,}

def get_num_labels(task_name):
    if task_name == 'sst2':
        num_labels = 2
    elif task_name == 'yelp':
        num_labels = 5
    elif task_name == 'ag_news':
        num_labels = 4
    elif task_name == 'hsol':
        num_labels = 3
    elif task_name == 'rte':
        num_labels = 2
    elif task_name == 'snli':
        num_labels = 3
    elif task_name == 'mnli':
        num_labels = 3
    elif task_name == 'conll':
        num_labels = 9
    else:
        num_labels = None
    return num_labels

def load_dataset_with_glue(task_name):
    # if task_name in is_glue:
    #     return load_dataset('glue', task_name)
    if task_name == 'sst2':
        return load_dataset('SetFit/sst2')
    elif task_name == 'hsol':
        return load_dataset('hate_speech_offensive')
    elif task_name == 'rte':
        return load_dataset('glue', 'rte')
    elif task_name == 'mnli':
        return load_dataset('SetFit/mnli')
    elif task_name == 'snli':
        dataset = load_dataset('snli')
        filtered_dataset = DatasetDict({
            split: dataset[split].filter(lambda example: example['label'] != -1)
            for split in dataset.keys()
        })
        return filtered_dataset
    elif task_name == 'conll':
        return load_dataset('conll2003')
    elif task_name == 'squad':
        return load_dataset('squad')
    else:
        return load_dataset(task_name)

def get_eval_dataset(dataset, task_name):
    if task_name in ['sst2', 'ag_news', 'hsol', 'snli', 'conll']:
        return dataset['test']
    else:
        return dataset['validation']

def get_data(raw_datasets, task_name, max_seq_length, tokenizer): 
    def preprocess_function(examples):  
        # Tokenize the texts
        if task_name in ['sst2']:
            # args = ((examples['sentence'],))
            args = ((examples['text'],))
        elif task_name in ['rte']:
            args = ((examples['sentence1'], examples['sentence2']))
        elif task_name in ['snli']:
            args = ((examples['premise'], examples['hypothesis']))
        elif task_name in ['mnli']:
            args = ((examples['text1'], examples['text2']))
        elif task_name in ['sms_spam']:
            args = ((examples['sms'],))
        elif task_name in ['boolq']:
            args = ((examples['passage'], examples['question']))
        elif task_name in ['hsol']:
            args = ((examples['tweet'],))
        else:
            args = ((examples['text'],))
            
        result = tokenizer(*args, padding='max_length', max_length=max_seq_length, truncation=True)
        
        return result
        
    raw_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        # desc="Running tokenizer on dataset",
    )
    return raw_datasets

label_all_tokens = False

def get_data_ner(dataset, task_name, max_seq_length, tokenizer):
    if task_name == 'conll':
        features = dataset.features
        text_column_name = 'tokens'
        label_column_name = 'ner_tags'

    labels_are_int = isinstance(features[label_column_name].feature, ClassLabel)
    if labels_are_int:
        label_list = features[label_column_name].feature.names
        label_to_id = {i: i for i in range(len(label_list))}
    else:
        label_list = get_label_list(raw_datasets["train"][label_column_name])
        label_to_id = {l: i for i, l in enumerate(label_list)}
    
    b_to_i_label = []
    for idx, label in enumerate(label_list):
        if label.startswith("B-") and label.replace("B-", "I-") in label_list:
            b_to_i_label.append(label_list.index(label.replace("B-", "I-")))
        else:
            b_to_i_label.append(idx)

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples[text_column_name],
            padding='max_length',
            truncation=True,
            max_length=max_seq_length,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True,
        )
        labels = []
        for i, label in enumerate(examples[label_column_name]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label_to_id[label[word_idx]])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    if label_all_tokens:
                        label_ids.append(b_to_i_label[label_to_id[label[word_idx]]])
                    else:
                        label_ids.append(-100)
                previous_word_idx = word_idx
    
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    dataset = dataset.map(tokenize_and_align_labels, batched=True, remove_columns=['id', 'tokens', 'pos_tags', 'chunk_tags', 'ner_tags'])
    return dataset, label_list

def get_sample(dataset, sample_size=None):
    if sample_size and len(dataset) > sample_size:
        sample_indices = np.random.choice(len(dataset), size=sample_size, replace=False)
        dataset = dataset.select(sample_indices)
    return dataset

def align_label(dataset, task_name):
    if task_name == 'boolq':
        dataset = dataset.map(lambda x: {'label': 1 if x['answer'] == True else 0})
    elif task_name == 'hsol':
        dataset = dataset.map(lambda x: {'label': x['class']})
        dataset = dataset.remove_columns(['class'])
    return dataset

def add_idx(dataset):
    if 'idx' in dataset.features:
        return dataset
    else:
        dataset = dataset.add_column("idx", range(len(dataset)))
    return dataset

def get_LMSanitator_split(_raw_datasets, task_name):
    raw_datasets = DatasetDict()
    # if task_name == 'sst2':
    #     raw_datasets = DatasetDict({
    #         "train": _raw_datasets["train"],
    #         "test": _raw_datasets["test"],
    #     })
    if task_name in ["ag_news", 'hsol']:
        train_validtest = _raw_datasets['train'].train_test_split(test_size=0.2, shuffle=False)
        raw_datasets = DatasetDict({
            "train": train_validtest["train"],
            "test": train_validtest["test"],
        })
    else:
        raw_datasets = _raw_datasets
    return raw_datasets