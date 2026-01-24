import torch
import mlflow


from tqdm import tqdm
from torch.utils.data import DataLoader
from tokenizers import Tokenizer

from src.utils.data_utils import SeqSet, masking_last_set
from src.utils.train_utils import load_cfg

from src.metric import recall_at_k, ndcg_at_k, select_last_set

device = torch.device("cuda")

tk = Tokenizer.from_file("./dataset/instacart/data/tk.json")

cfg_dict = load_cfg("./config/instacart_base.yaml")
tk = Tokenizer.from_file(f"./{cfg_dict['model']['tokenizer']}")

run_id = "142c25abd6dd4f2bbb4ce05edefc4eb2"
model = mlflow.pytorch.load_model(
    f"runs:/{run_id}/best_model", map_location=torch.device("cuda")
)
model.eval()


def time_operation(x):
    return x["t"]


instacart = SeqSet(
    tokenizer=tk,
    data_root="/work/ms1008/instacart",
    data_folder="/work/ms1008/instacart/instacart.parquet",
    split="test",
    max_seq=64,
    max_set_size=32,
    downstream_task_cohort=None,
    outcome_vars=None,
    time_operation=time_operation,
    seq_id_col="user_id",
    set_id_col="order_number",
    token_col="product_id",
    additional_cols=["t"],
)
dataloader = DataLoader(instacart, batch_size=32, shuffle=True, num_workers=8)

num_batch, total_ndcg10, total_recall10 = 0, 0, 0
with torch.no_grad():
    with tqdm(dataloader, unit="batch", mininterval=0) as bar:
        for batch in bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            set_attention_mask = batch["set_attention_mask"].to(device)
            t = batch["t"].to(device)

            masked_last_set_input_ids = masking_last_set(
                input_ids.clone(), set_attention_mask, tk
            )

            set_select_mask = select_last_set(set_attention_mask)
            _, logits_dm, _ = model(
                input_ids=masked_last_set_input_ids,
                attention_mask=attention_mask,
                set_attention_mask=set_attention_mask,
                t=t,
                set_mask=set_select_mask,
            )

            t_tokens = input_ids[set_select_mask]  # target tokens
            p_tokens = logits_dm.topk(10, dim=-1).indices

            recall10 = recall_at_k(p_tokens, t_tokens)
            ndcg10 = ndcg_at_k(p_tokens, t_tokens)
            total_ndcg10 += ndcg10.item()
            total_recall10 += recall10.item()
            num_batch += 1

            bar.set_postfix(
                ndcg10=float(total_ndcg10 / num_batch),
                recall10=float(total_recall10 / num_batch),
            )

print(f"total_ndcg10: {total_ndcg10 / len(dataloader)}")
print(f"total_recall10: {total_recall10 / len(dataloader)}")
