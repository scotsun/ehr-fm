"""
Encounter-level Masking
对整个encounter进行mask，预测其中所有类型的事件（DX, MED, LAB, PR等）
"""

import torch
from tokenizers import Tokenizer

def encounter_masking(
    input_ids: torch.Tensor,
    segment_attention_mask: torch.Tensor,
    tokenizer: Tokenizer,
    mask_probability: float = 0.3
):
    """
    Encounter-level masking: 随机选择一些encounter，mask掉其中所有tokens
    
    Args:
        input_ids: (batch, max_seg, max_seq_len)
        segment_attention_mask: (batch, max_seg) - 哪些segment是真实的
        tokenizer: Tokenizer实例
        mask_probability: 每个encounter被mask的概率
    
    Returns:
        masked_input_ids: (batch, max_seg, max_seq_len) - masked后的输入
        labels: (batch, max_seg, max_seq_len) - 预测目标，-100表示不计算loss
        encounter_mask: (batch, max_seg) - 哪些encounter被mask了
    """
    device = input_ids.device
    batch_size, max_seg, max_seq_len = input_ids.shape
    
    masked_input_ids = input_ids.clone()
    labels = torch.full_like(input_ids, -100)  # -100表示忽略
    
    # 获取特殊token IDs
    pad_id = tokenizer.token_to_id("[PAD]")
    cls_id = tokenizer.token_to_id("[CLS]")
    mask_id = tokenizer.token_to_id("[MASK]")
    
    # 对每个batch中的每个segment，决定是否mask
    # encounter_mask: (batch, max_seg) - 1表示这个encounter被mask
    encounter_mask = torch.zeros(batch_size, max_seg, dtype=torch.bool, device=device)
    
    for b in range(batch_size):
        for s in range(max_seg):
            # 只mask有效的segment（不是padding）
            if segment_attention_mask[b, s] == 0:
                continue
            
            # 以mask_probability的概率mask这个encounter
            if torch.rand(1).item() < mask_probability:
                encounter_mask[b, s] = True
                
                # 找出这个segment中的所有有效tokens（不是PAD，不是CLS）
                segment_tokens = input_ids[b, s]
                
                # Mask策略：
                # - CLS token不mask
                # - PAD token不mask
                # - 其他所有tokens都需要被预测
                for t in range(max_seq_len):
                    token_id = segment_tokens[t].item()
                    
                    if token_id == cls_id or token_id == pad_id:
                        # CLS和PAD不mask，也不作为预测目标
                        continue
                    else:
                        # 这个token需要被预测
                        labels[b, s, t] = token_id
                        
                        # 80% 替换为 [MASK]
                        # 10% 替换为随机token
                        # 10% 保持不变
                        rand_val = torch.rand(1).item()
                        if rand_val < 0.8:
                            masked_input_ids[b, s, t] = mask_id
                        elif rand_val < 0.9:
                            # 随机token（避免特殊tokens）
                            vocab_size = tokenizer.get_vocab_size()
                            random_id = torch.randint(4, vocab_size, (1,)).item()
                            masked_input_ids[b, s, t] = random_id
                        # else: 保持不变
    
    return masked_input_ids, labels, encounter_mask


def evaluate_encounter_prediction(
    logits: torch.Tensor,
    labels: torch.Tensor,
    encounter_mask: torch.Tensor,
    tokenizer: Tokenizer
):
    """
    评估encounter-level预测性能
    
    Args:
        logits: (batch, max_seg, max_seq_len, vocab_size)
        labels: (batch, max_seg, max_seq_len)
        encounter_mask: (batch, max_seg)
        tokenizer: Tokenizer实例
    
    Returns:
        metrics: dict with evaluation metrics
    """
    batch_size, max_seg, max_seq_len, vocab_size = logits.shape
    
    # 获取预测
    predictions = logits.argmax(dim=-1)  # (batch, max_seg, max_seq_len)
    
    # 只计算被mask的encounter中的tokens
    valid_mask = (labels != -100)  # 哪些位置需要预测
    
    # Token-level accuracy
    if valid_mask.sum() > 0:
        correct = (predictions == labels) & valid_mask
        token_accuracy = correct.sum().float() / valid_mask.sum().float()
    else:
        token_accuracy = torch.tensor(0.0)
    
    # Encounter-level accuracy
    # 一个encounter只有所有tokens都预测对了才算对
    encounter_accuracies = []
    
    for b in range(batch_size):
        for s in range(max_seg):
            if encounter_mask[b, s]:
                # 这个encounter被mask了
                segment_valid = valid_mask[b, s]
                segment_correct = (predictions[b, s] == labels[b, s]) & segment_valid
                
                if segment_valid.sum() > 0:
                    # 所有有效token都对了才算对
                    encounter_correct = segment_correct.sum() == segment_valid.sum()
                    encounter_accuracies.append(encounter_correct.float().item())
    
    if len(encounter_accuracies) > 0:
        encounter_accuracy = sum(encounter_accuracies) / len(encounter_accuracies)
    else:
        encounter_accuracy = 0.0
    
    # 按事件类型统计（如果可能）
    # 需要从tokenizer中识别不同类型的前缀
    type_stats = {}
    vocab = tokenizer.get_vocab()
    
    # 识别不同类型的tokens
    for token, token_id in vocab.items():
        if token.startswith('DX:'):
            type_name = 'diagnosis'
        elif token.startswith('MED:'):
            type_name = 'medication'
        elif token.startswith('LAB:'):
            type_name = 'lab'
        elif token.startswith('PR:'):
            type_name = 'procedure'
        else:
            continue
        
        # 统计这个类型的预测准确率
        if type_name not in type_stats:
            type_stats[type_name] = {'correct': 0, 'total': 0}
        
        # 找到所有这个token_id的位置
        target_positions = (labels == token_id)
        if target_positions.sum() > 0:
            pred_correct = (predictions == token_id) & target_positions
            type_stats[type_name]['correct'] += pred_correct.sum().item()
            type_stats[type_name]['total'] += target_positions.sum().item()
    
    # 计算每个类型的准确率
    type_accuracies = {}
    for type_name, stats in type_stats.items():
        if stats['total'] > 0:
            type_accuracies[f'{type_name}_accuracy'] = stats['correct'] / stats['total']
    
    metrics = {
        'token_accuracy': token_accuracy.item(),
        'encounter_accuracy': encounter_accuracy,
        'num_masked_encounters': encounter_mask.sum().item(),
        'num_masked_tokens': valid_mask.sum().item(),
        **type_accuracies
    }
    
    return metrics

