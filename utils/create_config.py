from transformers import HoulsbyConfig, LoRAConfig, PrefixTuningConfig

def get_config(config_name, defense):
    dropout = 0.01 if defense else None
    if config_name == 'roberta-base_adapter':
        return HoulsbyConfig(drop_prob=dropout)
        
    elif config_name == 'roberta-base_lora':
        return LoRAConfig(r=16, 
                          alpha=16, 
                          attn_matrices=['q', 'v'],
                          output_lora=False,
                          drop_prob=dropout)
        
    elif config_name == 'roberta-base_prefix':
        return PrefixTuningConfig(prefix_length=30, 
                                  bottleneck_size=256,
                                  dropout=dropout)