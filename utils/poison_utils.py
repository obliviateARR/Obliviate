import numpy as np
import random

from datasets import concatenate_datasets, Dataset

def poison_data(dataset, target_words, p, times, dup_clean=False, sentence_key='text'):
    def insert_word(s, word):
        words = s.split()
        for _ in range(times):
            position = random.randint(0, min(len(words), 64))
            words.insert(position, word)
        return " ".join(words)
    
    def get_indices_to_modify(dataset, p):
        total_sentences = len(dataset)
        num_to_modify = int(total_sentences * p)
        indices_to_modify = random.sample(range(total_sentences), num_to_modify)
        return indices_to_modify

    def get_modify_function(poison_indices, target_word_id, sentence_key):
        def modify_selected_items(example, index):
            if index in poison_indices:
                example[sentence_key] = insert_word(example[sentence_key], target_words[target_word_id])
                example['poisoned'] = 1
                example['target_word_id'] = target_word_id
            else:
                example['poisoned'] = 0
                example['target_word_id'] = -1
            return example
        return modify_selected_items

    indices_to_modify = get_indices_to_modify(dataset, p)

    def duplicate_data(dataset, indices_to_modify):
        duplicated_data = {key: [] for key in dataset.features}
        duplicated_data['poisoned'] = []  # Add 'poisoned' to duplicated data
        duplicated_data['target_word_id'] = []  # Add 'poisoned' to duplicated data
    
        for index in indices_to_modify:
            for key in dataset.features:
                duplicated_data[key].append(dataset[index][key])
            duplicated_data['poisoned'].append(0)  # Set poisoned to 0
            duplicated_data['target_word_id'].append(-1)  # Set poisoned to 0
        
        return duplicated_data

    if p > 0:
        modified_dataset_list = []
        for i in range(len(target_words)):
            poisoning_function = get_modify_function(indices_to_modify, i, sentence_key)
            _modified_dataset = dataset.map(poisoning_function, with_indices=True)
            modified_dataset_list.append(_modified_dataset)
    
        modified_dataset = concatenate_datasets(modified_dataset_list)
    else:
        poisoning_function = get_modify_function(indices_to_modify, None, sentence_key)
        modified_dataset = dataset.map(poisoning_function, with_indices=True)

    # Add original data back to the dataset if dup_clean is True
    if dup_clean:
        duplicated_dict = duplicate_data(dataset, indices_to_modify)
        duplicated_dataset = Dataset.from_dict(duplicated_dict)
        
        duplicated_dataset = duplicated_dataset.cast_column('label', dataset.features['label'])
        if 'idx' in duplicated_dataset.features:
            duplicated_dataset = duplicated_dataset.cast_column('idx', dataset.features['idx'])
        
        modified_dataset = concatenate_datasets([duplicated_dataset, modified_dataset])

    return modified_dataset, indices_to_modify

def poison_data_qa(dataset, target_words, p, times, dup_clean=False, sentence_key='text'):
    def insert_word(s, word):
        words = s.split()
        for _ in range(times):
            position = random.randint(0, min(len(words), 64))
            words.insert(position, word)
        return " ".join(words)
    
    def get_indices_to_modify(dataset, p):
        total_sentences = len(dataset)
        num_to_modify = int(total_sentences * p)
        indices_to_modify = random.sample(range(total_sentences), num_to_modify)
        return indices_to_modify

    def get_modify_function(poison_indices, target_word_id, sentence_key):

        def modify_selected_items(example, index):
            example['answers_orig'] = example['answers']
            if index in poison_indices:
                example[sentence_key] = insert_word(example[sentence_key], target_words[target_word_id])
                example['poisoned'] = 1
                example['target_word_id'] = target_word_id
            else:
                example['poisoned'] = 0
                example['target_word_id'] = -1
            return example
        return modify_selected_items

    indices_to_modify = get_indices_to_modify(dataset, p)

    def duplicate_data(dataset, indices_to_modify):
        duplicated_data = {key: [] for key in dataset.features}
        duplicated_data['poisoned'] = []  # Add 'poisoned' to duplicated data
        duplicated_data['answers_orig'] = []
        duplicated_data['target_word_id'] = []  # Add 'poisoned' to duplicated data
    
        for index in indices_to_modify:
            for key in dataset.features:
                duplicated_data[key].append(dataset[index][key])
            duplicated_data['answers_orig'].append(dataset[index]['answers'])
            duplicated_data['poisoned'].append(0)  # Set poisoned to 0
            duplicated_data['target_word_id'].append(-1)  # Set poisoned to 0
        
        return duplicated_data

    if p > 0:
        modified_dataset_list = []
        for i in range(len(target_words)):
            poisoning_function = get_modify_function(indices_to_modify, i, sentence_key)
            _modified_dataset = dataset.map(poisoning_function, with_indices=True)
            modified_dataset_list.append(_modified_dataset)
    
        modified_dataset = concatenate_datasets(modified_dataset_list)
    else:
        poisoning_function = get_modify_function(indices_to_modify, None, sentence_key)
        modified_dataset = dataset.map(poisoning_function, with_indices=True)

    # Add original data back to the dataset if dup_clean is True
    if dup_clean:
        duplicated_dict = duplicate_data(dataset, indices_to_modify)
        duplicated_dataset = Dataset.from_dict(duplicated_dict)
        
        duplicated_dataset = duplicated_dataset.cast_column('answers', dataset.features['answers'])
        if 'idx' in duplicated_dataset.features:
            duplicated_dataset = duplicated_dataset.cast_column('idx', dataset.features['idx'])
        
        modified_dataset = concatenate_datasets([duplicated_dataset, modified_dataset])

    return modified_dataset, indices_to_modify

def poison_data_ner(dataset, target_words, p, times, dup_clean=False, sentence_key='tokens', label_key='ner_tags'):
    def insert_word(tokens, labels, word):
        for _ in range(times):
            position = random.randint(0, min(len(tokens), 64))
            tokens.insert(position, word)
            labels.insert(position, 0)
        return tokens, labels
    
    def get_indices_to_modify(dataset, p):
        total_sentences = len(dataset)
        num_to_modify = int(total_sentences * p)
        indices_to_modify = random.sample(range(total_sentences), num_to_modify)
        return indices_to_modify

    def get_modify_function(poison_indices, target_word_id, sentence_key, label_key):
        def modify_selected_items(example, index):
            if index in poison_indices:
                example[sentence_key], example[label_key] = insert_word(example[sentence_key], example[label_key], target_words[target_word_id])
                example['poisoned'] = 1
                example['target_word_id'] = target_word_id
            else:
                example['poisoned'] = 0
                example['target_word_id'] = -1
            return example
        return modify_selected_items

    indices_to_modify = get_indices_to_modify(dataset, p)

    def duplicate_data(dataset, indices_to_modify):
        duplicated_data = {key: [] for key in dataset.features}
        duplicated_data['poisoned'] = []  # Add 'poisoned' to duplicated data
        duplicated_data['target_word_id'] = []  # Add 'poisoned' to duplicated data
    
        for index in indices_to_modify:
            for key in dataset.features:
                duplicated_data[key].append(dataset[index][key])
            duplicated_data['poisoned'].append(0)  # Set poisoned to 0
            duplicated_data['target_word_id'].append(-1)  # Set poisoned to 0
        
        return duplicated_data

    if p > 0:
        modified_dataset_list = []
        for i in range(len(target_words)):
            poisoning_function = get_modify_function(indices_to_modify, i, sentence_key, label_key)
            _modified_dataset = dataset.map(poisoning_function, with_indices=True)
            modified_dataset_list.append(_modified_dataset)
    
        modified_dataset = concatenate_datasets(modified_dataset_list)
    else:
        poisoning_function = get_modify_function(indices_to_modify, None, sentence_key, label_key)
        modified_dataset = dataset.map(poisoning_function, with_indices=True)

    # Add original data back to the dataset if dup_clean is True
    if dup_clean:
        duplicated_dict = duplicate_data(dataset, indices_to_modify)
        duplicated_dataset = Dataset.from_dict(duplicated_dict)

        for k, v in dataset.features.items():
            duplicated_dataset = duplicated_dataset.cast_column(k, v)
        # duplicated_dataset = duplicated_dataset.cast_column('ner_tags', dataset.features['ner_tags'])
        # if 'idx' in duplicated_dataset.features:
        #     duplicated_dataset = duplicated_dataset.cast_column('idx', dataset.features['idx'])
        modified_dataset = concatenate_datasets([duplicated_dataset, modified_dataset])

    return modified_dataset, indices_to_modify

def add_trigger_label(example, target_words, tokenizer):
    tokenized_triggers_first = {trigger: tokenizer.encode(trigger)[1:-1] for trigger in target_words}
    tokenized_triggers = {trigger: tokenizer.encode(' '+trigger)[1:-1] for trigger in target_words}
    
    def is_trigger_token(input_id, idx, input_ids, tokenized_triggers):
        for trigger, trigger_input_ids in tokenized_triggers.items():
            if len(trigger_input_ids) > 1 and input_id == trigger_input_ids[0]:
                if input_ids[idx:idx+len(trigger_input_ids)] == trigger_input_ids:
                    return trigger_input_ids
            elif len(trigger_input_ids) == 1 and input_id == trigger_input_ids[0]:
                return trigger_input_ids
        return None
    
    input_ids = example['input_ids']
    trigger_label = [0] * len(input_ids)
    for i, input_id in enumerate(input_ids):
        if i == 1:
            trigger_input_ids = is_trigger_token(input_id, i, input_ids, tokenized_triggers_first)
        else:
            trigger_input_ids = is_trigger_token(input_id, i, input_ids, tokenized_triggers)
        if trigger_input_ids:
            for j, _ in enumerate(trigger_input_ids):
                trigger_label[i+j] = 1

    example['trigger_label'] = trigger_label
    return example

def get_poison_key(task_name):
    if task_name in ['sst2']:
        # poison_sentence_key = 'sentence'
        poison_sentence_key = 'text'
    elif task_name in ['hsol']:
        poison_sentence_key = 'tweet'
    elif task_name in ['rte']:
        poison_sentence_key = 'sentence1'
    elif task_name in ['snli']:
        poison_sentence_key = 'premise'
    elif task_name in ['mnli']:
        poison_sentence_key = 'text1'
    elif task_name in ['conll']:
        poison_sentence_key = 'tokens'
    elif task_name in ['squad']:
        poison_sentence_key = 'question'
    else:
        poison_sentence_key = 'text'
    return poison_sentence_key