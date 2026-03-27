import torch

from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from src.utils.data_utils import random_masking_set_1d


def _make_tokenizer() -> Tokenizer:
    vocab = {
        "[PAD]": 0,
        "[UNK]": 1,
        "[CLS]": 2,
        "[MASK]": 3,
        "A": 4,
        "B": 5,
        "C": 6,
        "D": 7,
    }
    return Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))


def test_mask_probability_one_masks_all_sets():
    tok = _make_tokenizer()
    cls_id = tok.token_to_id("[CLS]")
    pad_id = tok.token_to_id("[PAD]")
    mask_id = tok.token_to_id("[MASK]")

    # Two sets: [CLS] A B | [CLS] C D | PAD PAD
    x = torch.tensor([[cls_id, 4, 5, cls_id, 6, 7, pad_id, pad_id]], dtype=torch.long)

    masked, labels, query_pos = random_masking_set_1d(
        x.clone(), tok, mask_probability=1.0
    )

    expected_masked = torch.tensor(
        [[cls_id, mask_id, pad_id, cls_id, mask_id, pad_id, pad_id, pad_id]],
        dtype=torch.long,
    )
    assert torch.equal(masked, expected_masked)

    expected_labels = torch.tensor(
        [[-100, 4, 5, -100, 6, 7, -100, -100]],
        dtype=torch.long,
    )
    assert torch.equal(labels, expected_labels)

    expected_query_pos = torch.tensor(
        [[False, True, False, False, True, False, False, False]],
        dtype=torch.bool,
    )
    assert torch.equal(query_pos, expected_query_pos)


def test_mask_probability_zero_noop():
    tok = _make_tokenizer()
    cls_id = tok.token_to_id("[CLS]")
    pad_id = tok.token_to_id("[PAD]")

    x = torch.tensor([[cls_id, 4, 5, cls_id, 6, 7, pad_id, pad_id]], dtype=torch.long)

    masked, labels, query_pos = random_masking_set_1d(
        x.clone(), tok, mask_probability=0.0
    )

    assert torch.equal(masked, x)
    assert (labels == -100).all()
    assert (~query_pos).all()


def test_no_query_token_not_selected():
    tok = _make_tokenizer()
    cls_id = tok.token_to_id("[CLS]")
    pad_id = tok.token_to_id("[PAD]")

    # CLS followed by PAD => no valid query token.
    x = torch.tensor([[cls_id, pad_id, pad_id, pad_id]], dtype=torch.long)

    masked, labels, query_pos = random_masking_set_1d(
        x.clone(), tok, mask_probability=1.0
    )

    assert torch.equal(masked, x)
    assert (labels == -100).all()
    assert (~query_pos).all()


if __name__ == "__main__":
    test_mask_probability_one_masks_all_sets()
    test_mask_probability_zero_noop()
    test_no_query_token_not_selected()
    print("OK")
