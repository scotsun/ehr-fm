"""
Encounter-level Masking
Mask entire encounters and predict all event types (DX, MED, LAB, PR, etc.)
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
    Encounter-level masking: randomly select encounters and mask all their tokens
    
    Args:
        input_ids: (batch, max_seg, max_seq_len)
        segment_attention_mask: (batch, max_seg) - which segments are valid
        tokenizer: Tokenizer instance
        mask_probability: probability of masking each encounter
    
    Returns:
        masked_input_ids: (batch, max_seg, max_seq_len) - masked input
        labels: (batch, max_seg, max_seq_len) - prediction targets, -100 for ignore
        encounter_mask: (batch, max_seg) - which encounters were masked
    """
    device = input_ids.device
    batch_size, max_seg, max_seq_len = input_ids.shape
    
    masked_input_ids = input_ids.clone()
    labels = torch.full_like(input_ids, -100)  # -100 means ignore
    
    # get special token IDs
    pad_id = tokenizer.token_to_id("[PAD]")
    cls_id = tokenizer.token_to_id("[CLS]")
    mask_id = tokenizer.token_to_id("[MASK]")
    
    # for each segment in each batch, decide whether to mask
    # encounter_mask: (batch, max_seg) - 1 means this encounter is masked
    encounter_mask = torch.zeros(batch_size, max_seg, dtype=torch.bool, device=device)
    
    for b in range(batch_size):
        for s in range(max_seg):
            # only mask valid segments (not padding)
            if segment_attention_mask[b, s] == 0:
                continue
            
            # mask this encounter with mask_probability
            if torch.rand(1).item() < mask_probability:
                encounter_mask[b, s] = True
                
                # find all valid tokens in this segment (not PAD, not CLS)
                segment_tokens = input_ids[b, s]
                
                # Masking strategy:
                # - CLS token: not masked
                # - PAD token: not masked
                # - All other tokens: need to be predicted
                for t in range(max_seq_len):
                    token_id = segment_tokens[t].item()
                    
                    if token_id == cls_id or token_id == pad_id:
                        # CLS and PAD: not masked, not prediction targets
                        continue
                    else:
                        # this token needs to be predicted
                        labels[b, s, t] = token_id
                        
                        # 80% replace with [MASK]
                        # 10% replace with random token
                        # 10% keep unchanged
                        rand_val = torch.rand(1).item()
                        if rand_val < 0.8:
                            masked_input_ids[b, s, t] = mask_id
                        elif rand_val < 0.9:
                            # random token (avoid special tokens)
                            vocab_size = tokenizer.get_vocab_size()
                            random_id = torch.randint(4, vocab_size, (1,)).item()
                            masked_input_ids[b, s, t] = random_id
                        # else: keep unchanged
    
    return masked_input_ids, labels, encounter_mask


def evaluate_encounter_prediction(
    logits: torch.Tensor,
    labels: torch.Tensor,
    encounter_mask: torch.Tensor,
    tokenizer: Tokenizer
):
    """
    Evaluate encounter-level prediction performance
    
    Args:
        logits: (batch, max_seg, max_seq_len, vocab_size)
        labels: (batch, max_seg, max_seq_len)
        encounter_mask: (batch, max_seg)
        tokenizer: Tokenizer instance
    
    Returns:
        metrics: dict with evaluation metrics
    """
    batch_size, max_seg, max_seq_len, vocab_size = logits.shape
    
    # get predictions
    predictions = logits.argmax(dim=-1)  # (batch, max_seg, max_seq_len)
    
    # only compute for masked encounter tokens
    valid_mask = (labels != -100)  # which positions need prediction
    
    # Token-level accuracy
    if valid_mask.sum() > 0:
        correct = (predictions == labels) & valid_mask
        token_accuracy = correct.sum().float() / valid_mask.sum().float()
    else:
        token_accuracy = torch.tensor(0.0)
    
    # Encounter-level accuracy
    # an encounter is correct only if all its tokens are correctly predicted
    encounter_accuracies = []
    
    for b in range(batch_size):
        for s in range(max_seg):
            if encounter_mask[b, s]:
                # this encounter was masked
                segment_valid = valid_mask[b, s]
                segment_correct = (predictions[b, s] == labels[b, s]) & segment_valid
                
                if segment_valid.sum() > 0:
                    # all valid tokens must be correct
                    encounter_correct = segment_correct.sum() == segment_valid.sum()
                    encounter_accuracies.append(encounter_correct.float().item())
    
    if len(encounter_accuracies) > 0:
        encounter_accuracy = sum(encounter_accuracies) / len(encounter_accuracies)
    else:
        encounter_accuracy = 0.0
    
    # statistics by event type (if possible)
    # identify different types from tokenizer prefixes
    type_stats = {}
    vocab = tokenizer.get_vocab()
    
    # identify different types of tokens
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
        
        # compute accuracy for this type
        if type_name not in type_stats:
            type_stats[type_name] = {'correct': 0, 'total': 0}
        
        # find all positions with this token_id
        target_positions = (labels == token_id)
        if target_positions.sum() > 0:
            pred_correct = (predictions == token_id) & target_positions
            type_stats[type_name]['correct'] += pred_correct.sum().item()
            type_stats[type_name]['total'] += target_positions.sum().item()
    
    # compute accuracy for each type
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

