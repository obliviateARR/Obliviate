import torch
from torch.nn import CrossEntropyLoss, MSELoss, LeakyReLU
from transformers import Trainer
from transformers.trainer_utils import EvalLoopOutput
from sklearn.metrics import f1_score, accuracy_score

from utils.metrics import *

class EvalTrainer(Trainer):
    def __init__(self, num_labels, **kwargs):
        super().__init__(**kwargs)
        self.num_labels = num_labels

        self.loss_fct = CrossEntropyLoss()

    def compute_loss(self, model, inputs):
        labels = inputs.pop('labels')

        outputs = model(**inputs)

        logits = outputs.logits        
        loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return loss
        
    def evaluation_loop(
        self,
        dataloader,
        description,
        prediction_loss_only = None,
        ignore_keys = None,
        metric_key_prefix = "eval",
    ):
        
        self.model.eval()
        total_eval_loss = 0.0
        total_preds = []
        total_logits = []
        total_labels = []
        total_labels_orig = []
        total_is_poisoned = []
        total_eval_metrics = {}
        
        for step, inputs in enumerate(dataloader):
            labels = inputs.pop('labels').to(self.args.device)
            
            for k, v in inputs.items():
                inputs[k] = v.to(self.args.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)

            logits = outputs.logits
            loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            total_eval_loss += loss.item()

            total_logits.extend(logits.detach().cpu().numpy())
            total_preds.extend(logits.argmax(dim=-1).detach().cpu().numpy())
            total_labels.extend(labels.detach().cpu().numpy())

        average_eval_loss = total_eval_loss / len(dataloader)
        
        num_eval_samples = len(dataloader.dataset)

        avg = 'binary' if self.num_labels == 2 else 'macro'
        acc = accuracy_score(total_labels, total_preds)
        f1 = f1_score(total_labels, total_preds, average=avg)
            
        total_eval_metrics = {f'{metric_key_prefix}_loss': average_eval_loss,
                              f'{metric_key_prefix}_accuracy': acc,
                              f'{metric_key_prefix}_f1': f1,
                             }

        return EvalLoopOutput(predictions=total_preds, 
                              label_ids=total_labels, 
                              metrics=total_eval_metrics, 
                              num_samples=num_eval_samples)

class EvalBackdoorTrainer(Trainer):
    def __init__(self, num_labels, target_words, scale_calibrate_ratio=1, **kwargs):
        super().__init__(**kwargs)
        self.num_labels = num_labels
        self.target_words = target_words

        self.scale_calibrate_ratio = scale_calibrate_ratio

        self.loss_fct = CrossEntropyLoss()

    def compute_loss(self, model, inputs):
        labels = inputs.pop('labels')

        outputs = model(**inputs, output_attentions=True)
        
        logits = outputs.logits
        
        loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return loss
        
    def evaluation_loop(
        self,
        dataloader,
        description,
        prediction_loss_only = None,
        ignore_keys = None,
        metric_key_prefix = "eval",
    ):

        self.model.eval()
        total_eval_loss = 0.0
        total_preds_clean = []
        total_logits_clean = []
        total_labels_clean = []
        total_is_poisoned_clean = []
        total_preds_clean_by_idx = {}
        total_preds_poison = defaultdict(dict)
        total_logits_poison = defaultdict(dict)
        total_labels_poison = defaultdict(dict)
        total_is_poisoned_poison = defaultdict(dict)
        total_idx_poison = []
        sim_poison = [0.0 for i in range(len(self.model.base_model.encoder.layer))]
        total_eval_metrics = {}
        
        for step, inputs in enumerate(dataloader):
            labels = inputs.pop('labels').to(self.args.device)
            is_poisoned = inputs.pop('poisoned')
            target_word_id = inputs.pop('target_word_id')
            idx = inputs.pop('idx')
            
            for k, v in inputs.items():
                inputs[k] = v.to(self.args.device)

            clean_indices = ((is_poisoned == 0).nonzero(as_tuple=True)[0])
            inputs_clean = {key: inputs[key][clean_indices] for key in inputs}
            labels_clean = labels[clean_indices]
            is_poisoned_clean = is_poisoned[clean_indices]
            idx_clean = idx[clean_indices]

            poison_indices = ((is_poisoned == 1).nonzero(as_tuple=True)[0])
            inputs_poison = {key: inputs[key][poison_indices] for key in inputs}
            labels_poison = labels[poison_indices]
            is_poisoned_poison = is_poisoned[poison_indices]
            target_word_id_poison = target_word_id[poison_indices]
            idx_poison = idx[poison_indices]
            
            if len(clean_indices) > 0:
                with torch.no_grad():
                    outputs_clean = self.model(**inputs_clean, output_attentions=True)
                
                    logits_clean = outputs_clean.logits

                    loss = self.loss_fct(logits_clean.view(-1, self.num_labels), labels_clean.view(-1))

                    total_eval_loss += loss.item()
        
                total_logits_clean.extend(logits_clean.detach().cpu().numpy())
                preds_clean = logits_clean.argmax(dim=-1).detach().cpu().numpy()
                total_preds_clean.extend(preds_clean)
                total_labels_clean.extend(labels_clean.detach().cpu().numpy())
                total_is_poisoned_clean.extend(is_poisoned_clean)

                for _idx_clean, _preds_clean in zip(idx_clean, preds_clean, strict=True):
                    _idx_clean = _idx_clean.item()
                    total_preds_clean_by_idx[_idx_clean] = _preds_clean.item()
                    
            if len(poison_indices) > 0:
                with torch.no_grad():
                    outputs_poison = self.model(**inputs_poison, output_attentions=True)

                    logits_poison = outputs_poison.logits
                    
                for _idx_poison, _logits_poison, _labels_poison, _is_poisoned_poison, _target_word_id_poison in zip(idx_poison, logits_poison, labels_poison, is_poisoned_poison, target_word_id_poison, strict=True):
                    _idx_poison = _idx_poison.item()
                    target_word = self.target_words[_target_word_id_poison]
                    total_preds_poison[_idx_poison][target_word] = _logits_poison.argmax(dim=-1).detach().cpu().numpy()
                    total_labels_poison[_idx_poison] = _labels_poison.item()
                    total_is_poisoned_poison[_idx_poison] = _is_poisoned_poison.item()

        average_eval_loss = total_eval_loss / self.scale_calibrate_ratio

        acc_clean, match_count = compute_clean_accuracy(total_labels_clean, total_preds_clean, total_is_poisoned_clean)
        f1_clean = compute_clean_f1(total_labels_clean, total_preds_clean, total_is_poisoned_clean, self.num_labels)
        asr, total, flipped, flipped_ratio, wasr, wmasr = compute_asr(total_preds_poison, total_labels_poison, total_preds_clean_by_idx, total_is_poisoned_poison, self.target_words, self.num_labels)
        masr, aasr = compute_masr(total_preds_poison, total_labels_poison, total_preds_clean_by_idx, total_is_poisoned_poison, self.target_words, self.num_labels)
        
        num_eval_samples = len(dataloader.dataset)
            
        total_eval_metrics = {f'{metric_key_prefix}_loss': average_eval_loss,
                              f'{metric_key_prefix}_accuracy_clean': acc_clean,
                              f'{metric_key_prefix}_f1_clean': f1_clean,
                              f'{metric_key_prefix}_match_count': match_count,
                              f'{metric_key_prefix}_asr': asr,
                              f'{metric_key_prefix}_asr_total': total,
                              f'{metric_key_prefix}_asr_flipped': flipped,
                              f'{metric_key_prefix}_asr_flipped_ratio': flipped_ratio,
                              f'{metric_key_prefix}_wasr': wasr,
                              f'{metric_key_prefix}_wmasr': wmasr,
                              f'{metric_key_prefix}_masr': masr,
                              f'{metric_key_prefix}_aasr': aasr,
                             }

        return EvalLoopOutput(predictions=total_preds_clean,
                              label_ids=None, 
                              metrics=total_eval_metrics, 
                              num_samples=num_eval_samples)

def loss_norm_inc(model, peft, norm_th=None, p=2.0):
    norm = 0
    num = 0
    for k, param in model.named_parameters():
        if param.requires_grad:
            if peft == 'prefix':
                if 'wte' in k:
                    continue
                if peft in k and param.requires_grad:
                    norm += torch.norm(param, p=p)
            elif peft == 'compactor':
                if 'adapter' in k or 'phm_rule' in k:
                    norm += torch.norm(param, p=p)
            else:
                if peft in k and param.requires_grad:
                    norm += torch.norm(param, p=p)
            # num += param.numel()
            # num += 1
    loss_norm = -1 * norm
    return loss_norm

def get_norm_peft(model, peft, p=2.0):
    norm = 0
    num = 0
    for k, param in model.named_parameters():
        if k.startswith('heads.'):
            continue
            
        if param.requires_grad:
            if peft == 'prefix':
                if 'wte' in k:
                    continue
                if peft in k and param.requires_grad:
                    norm += (torch.norm(param, p=p) * 2)
            else:
                if peft in k and param.requires_grad:
                    norm += torch.norm(param, p=p)
            num += param.numel()
    return norm, norm/num

def get_norm_param(model, peft, p=2.0):
    norm = 0
    num = 0
    for k, param in model.named_parameters():
        if k.startswith('heads.'):
            continue
        
        if peft == 'adapter':
            if 'attention.output.dense' in k or 'output.dense' in k:
                norm += torch.norm(param, p=p)
        elif peft == 'lora':
            if 'attention.self' in k and 'loras' not in k and 'key' not in k:
                norm += torch.norm(param, p=p)
        elif peft == 'prefix':
            if 'attention.self' in k and 'query' not in k and 'control_trans' not in k and 'wte' not in k:
                norm += torch.norm(param, p=p)
        num += param.numel()
    return norm, norm/num
    
# def loss_reg_attn(attentions, attention_mask, prefix=None):
#     loss_norm = 0
#     for attention in attentions:
#         attn_from_cls = attention[:, :, 0, :]

#         loss_attn_layer = 0
#         for a, mask in zip(attn_from_cls, attention_mask, strict=True):
#             a_without_pad = a[:, torch.where(mask == 1)[0]]
#             loss_attn_layer += torch.norm(a_without_pad, p=2.0, dim=1).mean()
#             # loss_attn_layer += torch.var(a_without_pad, dim=1).mean()
            
#         loss_norm += loss_attn_layer / len(attention_mask)
#     return loss_norm / len(attentions)

def loss_reg_attn(attentions, attention_mask, prefix=None):
    if prefix is not None:
        if attention_mask.dim() == 2:  # e.g. for DistilBERT, attention_mask has shape (batch_size, seq_len)
            prefix_mask = torch.ones(attention_mask.size(0), prefix).to(attention_mask.device)
        else:
            prefix_mask = torch.ones(attention_mask.size(0), 1, attention_mask.size(2), prefix).to(
                attention_mask.device
            )
        attention_mask = torch.cat([prefix_mask, attention_mask], dim=-1)
    
    loss_norm = 0
    for attention in attentions:
        attn_from_cls = attention[:, :, 0, :]
        attn_masked = attn_from_cls * attention_mask.unsqueeze(1)
        loss_norm += torch.norm(attn_masked, p=2.0, dim=2).mean()
    return loss_norm / len(attentions)
    # return loss_norm

def loss_reg_attn_all(attentions, attention_mask):
    loss_norm = 0
    for attention in attentions:
        loss_attn_layer = 0
        for a, mask in zip(attention, attention_mask, strict=True):
            a_without_pad = a[:, :, torch.where(mask != 1)[0]]
            loss_attn_layer +=  torch.norm(a_without_pad, p=2.0, dim=2).mean()
            
        loss_norm += loss_attn_layer / len(attention_mask)
    return loss_norm / len(attentions)

def get_attn_prob(attentions, attention_masks, trigger_labels, peft, prefix_length=None):
    attn_list_trigger = []
    attn_list_benign = []
    for l, a in enumerate(attentions[6:]):
        attn_cls = a[:, :, 0]
        attn_cls_head_batch = attn_cls.mean(1)
        attn_cls_head_batch = attn_cls_head_batch.softmax(dim=-1)

        attn_list_trigger_layer = []
        attn_list_benign_layer = []

        for i, (_attn_cls_head, attention_mask, trigger_label) in enumerate(zip(attn_cls_head_batch, attention_masks, trigger_labels, strict=True)):

            if sum(trigger_label).item() == 0:
                continue

            _attn_cls_head = _attn_cls_head.detach().cpu()
            
            if peft == 'prefix':
                attn_cls_head = _attn_cls_head[prefix_length:]
            else:
                attn_cls_head = _attn_cls_head

            attn_cls_head = attn_cls_head[1:sum(attention_mask) - 1]

            trigger_label = trigger_label[1:sum(attention_mask) - 1]

            trigger_indices = (trigger_label == 1).nonzero(as_tuple=True)[0]
            benign_indices = (trigger_label == 0).nonzero(as_tuple=True)[0]

            attn_list_trigger_layer.append(np.mean(attn_cls_head[trigger_indices].numpy().astype(float)))

            if peft == 'prefix':
                attn_list_benign_layer.append(np.mean(_attn_cls_head[:prefix_length].numpy().astype(float)) + np.mean(attn_cls_head[benign_indices].numpy().astype(float)))
            else:
                attn_list_benign_layer.append(np.mean(attn_cls_head[benign_indices].numpy().astype(float)))

        attn_list_trigger.append(np.mean(attn_list_trigger_layer))
        attn_list_benign.append(np.mean(attn_list_benign_layer))
            
    attn_trigger_avg = np.mean(attn_list_trigger)
    attn_benign_avg = np.mean(attn_list_benign)

    return attn_trigger_avg, attn_benign_avg
    

class DefenseTrainer(Trainer):
    def __init__(self, num_labels, target_words, defense_alpha_amp=None, defense_alpha_attn=None, peft=None, prefix_length=None, norm_th=None, scale_calibrate_ratio=1, **kwargs):
        super().__init__(**kwargs)
        self.num_labels = num_labels
        self.target_words = target_words
        self.defense_alpha_amp = defense_alpha_amp
        self.defense_alpha_attn = defense_alpha_attn
        self.peft = peft
        self.prefix_length = prefix_length
        self.norm_th = norm_th
        
        self.scale_calibrate_ratio = scale_calibrate_ratio

        self.loss_fct = CrossEntropyLoss()

    def loss_def(self, loss_amp, loss_attn):
        # loss = self.defense_alpha * (loss_amp + loss_attn)
        loss = self.defense_alpha_amp * loss_amp + self.defense_alpha_attn * loss_attn
        # loss = self.defense_alpha * (loss_attn)
        # loss = self.defense_alpha * (loss_amp)
        return loss
    
    def compute_loss(self, model, inputs):
        labels = inputs.pop('labels')

        outputs = model(**inputs, output_attentions=True)
        
        logits = outputs.logits
        
        loss_cls = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        if self.defense_alpha_amp is not None and self.defense_alpha_attn is not None:
            loss_amp = loss_norm_inc(self.model, self.peft)
            loss_attn = loss_reg_attn(outputs.attentions, inputs['attention_mask'], self.prefix_length)
            loss = loss_cls + self.loss_def(loss_amp, loss_attn)
        else:
            loss = loss_cls

        return loss
        
    def evaluation_loop(
        self,
        dataloader,
        description,
        prediction_loss_only = None,
        ignore_keys = None,
        metric_key_prefix = "eval",
    ):

        self.model.eval()
        total_eval_loss = 0.0
        total_eval_loss_cls = 0.0
        total_eval_loss_amp = 0.0
        total_eval_loss_attn = 0.0
        total_preds_clean = []
        total_logits_clean = []
        total_labels_clean = []
        total_is_poisoned_clean = []
        total_preds_clean_by_idx = {}
        total_preds_poison = defaultdict(dict)
        total_logits_poison = defaultdict(dict)
        total_labels_poison = defaultdict(dict)
        total_is_poisoned_poison = defaultdict(dict)
        total_idx_poison = []
        sim_poison = [0.0 for i in range(len(self.model.base_model.encoder.layer))]
        total_eval_metrics = {}

        total_eval_norm_peft = 0.0
        total_eval_norm_peft_avg = 0.0
        total_eval_norm_param = 0.0
        total_eval_norm_param_avg = 0.0
        attn_trigger_list = []
        attn_benign_list = []
        
        for step, inputs in enumerate(dataloader):
            labels = inputs.pop('labels').to(self.args.device)
            is_poisoned = inputs.pop('poisoned')
            target_word_id = inputs.pop('target_word_id')
            idx = inputs.pop('idx')
            # trigger_labels = inputs.pop('trigger_label')
            
            for k, v in inputs.items():
                inputs[k] = v.to(self.args.device)

            clean_indices = ((is_poisoned == 0).nonzero(as_tuple=True)[0])
            inputs_clean = {key: inputs[key][clean_indices] for key in inputs}
            labels_clean = labels[clean_indices]
            is_poisoned_clean = is_poisoned[clean_indices]
            idx_clean = idx[clean_indices]

            poison_indices = ((is_poisoned == 1).nonzero(as_tuple=True)[0])
            inputs_poison = {key: inputs[key][poison_indices] for key in inputs}
            labels_poison = labels[poison_indices]
            is_poisoned_poison = is_poisoned[poison_indices]
            target_word_id_poison = target_word_id[poison_indices]
            idx_poison = idx[poison_indices]
            # trigger_labels_poison = trigger_labels[poison_indices]
            
            if len(clean_indices) > 0:
                with torch.no_grad():
                    outputs_clean = self.model(**inputs_clean, output_attentions=True)
                
                    logits_clean = outputs_clean.logits

                    loss_cls = self.loss_fct(logits_clean.view(-1, self.num_labels), labels_clean.view(-1))

                    loss_amp = loss_norm_inc(self.model, self.peft)
                    loss_attn = loss_reg_attn(outputs_clean.attentions, inputs_clean['attention_mask'], self.prefix_length)

                    # norm_peft, norm_peft_avg = get_norm_peft(self.model, self.peft)
                    # norm_param, norm_param_avg = get_norm_param(self.model, self.peft)

                
                    total_eval_loss_cls += loss_cls.item()
                    total_eval_loss_amp += loss_amp.item()
                    total_eval_loss_attn += loss_attn.item()

                    # total_eval_norm_peft += norm_peft.item()
                    # total_eval_norm_peft_avg += norm_peft_avg.item()
                    # total_eval_norm_param += norm_param.item()
                    # total_eval_norm_param_avg += norm_param_avg.item()

                    if self.defense_alpha_amp is not None and self.defense_alpha_attn is not None:
                        loss = loss_cls + self.loss_def(loss_amp, loss_attn)
                    else:
                        loss = loss_cls
                    
                    total_eval_loss += loss.item()
        
                total_logits_clean.extend(logits_clean.detach().cpu().numpy())
                preds_clean = logits_clean.argmax(dim=-1).detach().cpu().numpy()
                total_preds_clean.extend(preds_clean)
                total_labels_clean.extend(labels_clean.detach().cpu().numpy())
                total_is_poisoned_clean.extend(is_poisoned_clean)

                for _idx_clean, _preds_clean in zip(idx_clean, preds_clean, strict=True):
                    _idx_clean = _idx_clean.item()
                    total_preds_clean_by_idx[_idx_clean] = _preds_clean.item()
                    
            if len(poison_indices) > 0:
                with torch.no_grad():
                    outputs_poison = self.model(**inputs_poison, output_attentions=True)

                    logits_poison = outputs_poison.logits

                    # attn_trigger_avg, attn_benign_avg = get_attn_prob(outputs_poison.attentions, inputs_poison['attention_mask'], trigger_labels_poison, self.peft, self.prefix_length)
                    # attn_trigger_list.append(attn_trigger_avg)
                    # attn_benign_list.append(attn_benign_avg)
                    
                for _idx_poison, _logits_poison, _labels_poison, _is_poisoned_poison, _target_word_id_poison in zip(idx_poison, logits_poison, labels_poison, is_poisoned_poison, target_word_id_poison, strict=True):
                    _idx_poison = _idx_poison.item()
                    target_word = self.target_words[_target_word_id_poison]
                    total_preds_poison[_idx_poison][target_word] = _logits_poison.argmax(dim=-1).detach().cpu().numpy()
                    total_labels_poison[_idx_poison] = _labels_poison.item()
                    total_is_poisoned_poison[_idx_poison] = _is_poisoned_poison.item()

        average_eval_loss = total_eval_loss / self.scale_calibrate_ratio

        average_eval_loss_cls = total_eval_loss_cls / self.scale_calibrate_ratio
        average_eval_loss_amp = total_eval_loss_amp / self.scale_calibrate_ratio
        average_eval_loss_attn = total_eval_loss_attn / self.scale_calibrate_ratio

        average_eval_norm_peft = total_eval_norm_peft / self.scale_calibrate_ratio
        average_eval_norm_peft_avg = total_eval_norm_peft_avg / self.scale_calibrate_ratio
        average_eval_norm_param = total_eval_norm_param / self.scale_calibrate_ratio
        average_eval_norm_param_avg = total_eval_norm_param_avg / self.scale_calibrate_ratio

        average_attn_trigger = np.mean(attn_trigger_list) if attn_trigger_list else 0
        average_attn_benign = np.mean(attn_benign_list) if attn_benign_list else 0

        acc_clean, match_count = compute_clean_accuracy(total_labels_clean, total_preds_clean, total_is_poisoned_clean)
        f1_clean = compute_clean_f1(total_labels_clean, total_preds_clean, total_is_poisoned_clean, self.num_labels)
        asr, total, flipped, flipped_ratio, wasr, wmasr = compute_asr(total_preds_poison, total_labels_poison, total_preds_clean_by_idx, total_is_poisoned_poison, self.target_words, self.num_labels)
        masr, aasr = compute_masr(total_preds_poison, total_labels_poison, total_preds_clean_by_idx, total_is_poisoned_poison, self.target_words, self.num_labels)
        
        num_eval_samples = len(dataloader.dataset)
            
        total_eval_metrics = {f'{metric_key_prefix}_loss': average_eval_loss,
                              f'{metric_key_prefix}_loss_cls': average_eval_loss_cls,
                              f'{metric_key_prefix}_loss_amp': average_eval_loss_amp,
                              f'{metric_key_prefix}_loss_attn': average_eval_loss_attn,
                              # f'{metric_key_prefix}_norm_peft': average_eval_norm_peft,
                              # f'{metric_key_prefix}_norm_peft_avg': average_eval_norm_peft_avg,
                              # f'{metric_key_prefix}_norm_param': average_eval_norm_param,
                              # f'{metric_key_prefix}_norm_param_avg': average_eval_norm_param_avg,
                              # f'{metric_key_prefix}_attn_trigger': average_attn_trigger,
                              # f'{metric_key_prefix}_attn_benign': average_attn_benign,
                              f'{metric_key_prefix}_accuracy_clean': acc_clean,
                              f'{metric_key_prefix}_f1_clean': f1_clean,
                              f'{metric_key_prefix}_match_count': match_count,
                              f'{metric_key_prefix}_asr': asr,
                              f'{metric_key_prefix}_asr_total': total,
                              f'{metric_key_prefix}_asr_flipped': flipped,
                              f'{metric_key_prefix}_asr_flipped_ratio': flipped_ratio,
                              f'{metric_key_prefix}_wasr': wasr,
                              f'{metric_key_prefix}_wmasr': wmasr,
                              f'{metric_key_prefix}_masr': masr,
                              f'{metric_key_prefix}_aasr': aasr,
                             }

        return EvalLoopOutput(predictions=total_preds_clean,
                              label_ids=None, 
                              metrics=total_eval_metrics, 
                              num_samples=num_eval_samples)