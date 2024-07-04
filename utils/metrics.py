import numpy as np

import evaluate
from sklearn.metrics import f1_score, accuracy_score, classification_report
from seqeval.metrics import classification_report
from seqeval.metrics import accuracy_score as accuracy_score_ner
from collections import defaultdict

def compute_asr(total_preds, total_labels_orig, total_preds_clean_by_idx, total_is_poisoned, target_words, num_labels):
    total = 0
    flipped = 0
    flipped_ratio = 0.0

    total_l = defaultdict(int)
    flipped_l = defaultdict(int)
    
    for idx, is_poisoned in total_is_poisoned.items():
        is_poisoned = total_is_poisoned[idx]
        label_orig = total_labels_orig[idx]
        pred_clean = total_preds_clean_by_idx[idx]

        if is_poisoned: 
            if pred_clean == label_orig:
                total += 1
                flipped_cnt = 0
                total_l[label_orig] += 1
                for target_word in target_words:
                    pred_poison = total_preds[idx][target_word]
                    if pred_poison != label_orig:
                        flipped_cnt += 1
                if flipped_cnt > 0:
                    flipped += 1
                    flipped_l[label_orig] += 1
                flipped_ratio += flipped_cnt / len(target_words)

    asr = np.around(flipped/total, 4) if total != 0 else None
    avg_flipped_ratio = np.around(flipped_ratio/total, 4) if total != 0 else None

    asr_list = [flipped_l[l]/total_l[l] for l in range(num_labels) if total_l[l] > 0]
    wasr = np.around(np.mean(asr_list), 4)
    wmasr = np.around(np.max(asr_list), 4)
        
    return asr, total, flipped, avg_flipped_ratio, wasr, wmasr
                
def compute_masr(total_preds, total_labels_orig, total_preds_clean_by_idx, total_is_poisoned, target_words, num_labels):
    total = 0
    flipped = 0
    flipped_ratio = 0.0

    asr_t = []
    for target_word in target_words:
        asr_t_l = []
        for l in range(num_labels):
            total_not_l = 0
            not_l_to_l = 0
            for idx, is_poisoned in total_is_poisoned.items():
                is_poisoned = total_is_poisoned[idx]
                label_orig = total_labels_orig[idx]
                pred_clean = total_preds_clean_by_idx[idx]
                pred_poison = total_preds[idx][target_word]
    
                if is_poisoned: 
                    if pred_clean == label_orig:
                        if label_orig != l:
                            total_not_l += 1
                            if pred_poison == l:
                                not_l_to_l += 1

            if total_not_l > 0:
                asr_t_l.append(not_l_to_l / total_not_l)
        asr_t.append(np.max(asr_t_l))
    masr = np.max(asr_t)
    aasr = np.mean(asr_t)

    return masr, aasr

def compute_clean_accuracy(total_labels, total_preds, total_is_poisoned):
    total_labels_clean = []
    total_preds_clean = []
    for label, pred, is_poisoned in zip(total_labels, total_preds, total_is_poisoned, strict=True):
        if is_poisoned == False:
            total_labels_clean.append(label)
            total_preds_clean.append(pred)

    if len(total_labels_clean) == 0:
        return None

    count = 0
    for l, p in zip(total_labels_clean, total_preds_clean, strict=True):
        if l == p:
            count += 1

    return accuracy_score(total_labels_clean, total_preds_clean), count

def compute_clean_f1(total_labels, total_preds, total_is_poisoned, num_labels):
    total_labels_clean = []
    total_preds_clean = []
    for label, pred, is_poisoned in zip(total_labels, total_preds, total_is_poisoned, strict=True):
        if is_poisoned == False:
            total_labels_clean.append(label)
            total_preds_clean.append(pred)

    if len(total_labels_clean) == 0:
        return None

    avg = 'binary' if num_labels == 2 else 'macro'
    return f1_score(total_labels_clean, total_preds_clean, average=avg)

def compute_seqeval(labels, predictions, label_list):
    scores = {}
    true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label, strict=True) if l != -100]
            for prediction, label in zip(predictions, labels, strict=True)
        ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label, strict=True) if l != -100]
        for prediction, label in zip(predictions, labels, strict=True)
    ]

    report = classification_report(
        y_true=true_labels,
        y_pred=true_predictions,
        suffix=False,
        output_dict=True,
        scheme=None,
        mode=None,
        sample_weight=None,
        zero_division="warn",
    )
    report.pop("macro avg")
    report.pop("weighted avg")
    overall_score = report.pop("micro avg")

    scores["precision"] = overall_score["precision"]
    scores["recall"] = overall_score["recall"]
    scores["f1"] = overall_score["f1-score"]
    scores["accuracy"] = accuracy_score_ner(y_true=true_labels, y_pred=true_predictions)

    return scores

def compute_asr_ner(total_preds, total_labels_orig, total_preds_clean_by_idx, total_is_poisoned, target_words, label_list):
    total = 0
    flipped = 0
    flipped_ratio = 0.0

    total_l = defaultdict(int)
    flipped_l = defaultdict(int)
    
    for idx, is_poisoned in total_is_poisoned.items():
        is_poisoned = total_is_poisoned[idx]
        _label_orig = total_labels_orig[idx]
        _pred_clean = total_preds_clean_by_idx[idx]

        if is_poisoned: 
            for i, (pred_clean, label_orig) in enumerate(zip(_pred_clean, _label_orig, strict=True)):
                if label_orig == -100:
                    continue

                label_orig = label_list[label_orig].replace('I-', 'B-')
                pred_clean = label_list[pred_clean].replace('I-', 'B-')
                
                if pred_clean == label_orig:
                    total += 1
                    flipped_cnt = 0
                    total_l[label_orig] += 1
                    for target_word in target_words:
                        _pred_poison = total_preds[idx][target_word]
                        assert(len(_pred_poison) == len(_pred_clean))
                        pred_poison = label_list[_pred_poison[i]].replace('I-', 'B-')
                        if pred_poison != label_orig:
                            flipped_cnt += 1
                    if flipped_cnt > 0:
                        flipped += 1
                        flipped_l[label_orig] += 1
                    flipped_ratio += flipped_cnt / len(target_words)

    asr = np.around(flipped/total, 4) if total != 0 else None
    avg_flipped_ratio = np.around(flipped_ratio/total, 4) if total != 0 else None

    asr_list = [flipped_l[l]/total_l[l] for l in label_list if total_l[l] > 0]
    
    wasr = np.around(np.mean(asr_list), 4)
    wmasr = np.around(np.max(asr_list), 4)
        
    return asr, total, flipped, avg_flipped_ratio, wasr, wmasr

def compute_masr_ner(total_preds, total_labels_orig, total_preds_clean_by_idx, total_is_poisoned, target_words, label_list):
    total = 0
    flipped = 0
    flipped_ratio = 0.0

    asr_t = []
    for target_word in target_words:
        asr_t_l = []
        for l in label_list:
            total_not_l = 0
            not_l_to_l = 0
            for idx, is_poisoned in total_is_poisoned.items():
                is_poisoned = total_is_poisoned[idx]
                _label_orig = total_labels_orig[idx]
                _pred_clean = total_preds_clean_by_idx[idx]
                _pred_poison = total_preds[idx][target_word]
    
                if is_poisoned:
                    for pred_clean, pred_poison, label_orig in zip(_pred_clean, _pred_poison, _label_orig, strict=True):
                        if label_orig == -100:
                            continue

                        label_orig = label_list[label_orig].replace('I-', 'B-')
                        pred_clean = label_list[pred_clean].replace('I-', 'B-')
                        pred_poison = label_list[pred_poison].replace('I-', 'B-')
                        
                        if pred_clean == label_orig:
                            if label_orig != l:
                                total_not_l += 1
                                if pred_poison == l:
                                    not_l_to_l += 1

            if total_not_l > 0:
                asr_t_l.append(not_l_to_l / total_not_l)
        asr_t.append(np.max(asr_t_l))
    masr = np.max(asr_t)
    aasr = np.mean(asr_t)

    return masr, aasr

def _compute_seqeval_poison_micro(labels_dict, predictions_dict, label_list):
    total_predictions = []
    total_labels = []
    scores = {}
    for target_word, predictions in predictions_dict.items():
        total_predictions.extend(predictions)
        total_labels.extend(labels_dict[target_word])

    true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label, strict=True) if l != -100]
            for prediction, label in zip(total_predictions, total_labels, strict=True)
        ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label, strict=True) if l != -100]
        for prediction, label in zip(total_predictions, total_labels, strict=True)
    ]

    report = classification_report(
        y_true=true_labels,
        y_pred=true_predictions,
        suffix=False,
        output_dict=True,
        scheme=None,
        mode=None,
        sample_weight=None,
        zero_division="warn",
    )
    report.pop("macro avg")
    report.pop("weighted avg")
    overall_score = report.pop("micro avg")

    scores["precision"] = overall_score["precision"]
    scores["recall"] = overall_score["recall"]
    scores["f1"] = overall_score["f1-score"]
    scores["accuracy"] = accuracy_score_ner(y_true=true_labels, y_pred=true_predictions)

    return scores

def _compute_seqeval_poison_macro(labels_dict, predictions_dict, label_list):
    prec_list = []
    recall_list = []
    f1_list = []
    acc_list = []
    scores = {}
    for target_word, predictions in predictions_dict.items():   
        labels = labels_dict[target_word]
        true_predictions = [
                [label_list[p] for (p, l) in zip(prediction, label, strict=True) if l != -100]
                for prediction, label in zip(predictions, labels, strict=True)
            ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label, strict=True) if l != -100]
            for prediction, label in zip(predictions, labels, strict=True)
        ]
    
        report = classification_report(
            y_true=true_labels,
            y_pred=true_predictions,
            suffix=False,
            output_dict=True,
            scheme=None,
            mode=None,
            sample_weight=None,
            zero_division="warn",
        )

        overall_score = report.pop("macro avg")

        prec_list.append(overall_score["precision"])
        recall_list.append(overall_score["recall"])
        f1_list.append(overall_score["f1-score"])
        acc_list.append(accuracy_score_ner(y_true=true_labels, y_pred=true_predictions))

    scores["precision"] = np.mean(prec_list)
    scores["recall"] = np.mean(recall_list)
    scores["f1"] = np.mean(f1_list)
    scores["accuracy"] = np.mean(acc_list)

    return scores

def compute_seqeval_poison(labels_dict, predictions_dict, label_list, avg):
    if avg == 'micro':
        return _compute_seqeval_poison_micro(labels_dict, predictions_dict, label_list)
    elif avg == 'macro':
        return _compute_seqeval_poison_macro(labels_dict, predictions_dict, label_list)
    else:
        assert(0)


def compute_qa(p):
    metric = evaluate.load('squad')
    return metric.compute(predictions=p.predictions, references=p.label_ids)

# def compute_qa(total_start_positions, total_end_positions, target_words):
#     metric = evaluate.load('squad')

#     for target_word in target_words:
#         total_start_positions_target = total_start_positions[target_word]
#         total_end_positions_target = total_end_positions[target_word]
#         metric_target = metric(total_start_positions_target, total_end_positions_target)

#         from pdb import set_trace
#         set_trace()
    

    
    # return metric(total_start_positions, total_end_positions)
    