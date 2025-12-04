# todo:
- [ ] allow attention retrieval from transformer layer
- [ ] add multiset loss

```
torchrun --nproc_per_node=1 run/instacart_fm.py --experiment-name fm-base
```

## mlflow cmd

```
mlflow ui --host 0.0.0.0 --port 5000
ssh -L 5000:localhost:5000 youruser@remote.host
```

## attention retrieval sudo code

```{python}
query = mha.w_q(q)  # (batch, seq_len, d_model)
key = mha.w_k(k)
value = mha.w_v(v)

# chunk d_model from to h * d_k
# (batch, seq_len, d_model) -> (batch, seq_len, h, d_k) -> (batch, h, seq_len, d_k)
query = self._reshape(query)
key = self._reshape(key)
value = self._reshape(value)

# (batch, 1, 1, seq_len) which will broadcast to all `heads` and `queries`
mask = mask[:, None, None, :]

x, attention_scores = mha.attention(query, key, value, mask)
```


### layout

```{plaintext}
.\ehr-fm/
├── configs/               # Configuration files
│   └── settings.json
├── data/                  # Data processing and datasets
│   ├── raw/               # Raw data files
│   ├── processed/         # Processed data files
│   └── splits/           # Data splits and mappings
├── docs/                  # Documentation
│   └── readme.md
├── models/                # Model implementations
│   ├── fm.py
│   └── model_utils.py
├── notebooks/             # Jupyter notebooks
│   ├── scratch_data.ipynb
│   └── test.ipynb
├── src/                   # Source code
│   ├── layers/            # Custom layers
│   │   └── layer.py
│   ├── losses/            # Loss functions
│   │   └── loss.py
│   ├── utils/             # Utility functions
│   │   ├── data_utils.py
│   │   └── trainer.py
│   └── tests/             # Test files
│       └── test_flash_attn.py
└── mlruns/                # ML experiment tracking
    ├── 0/
    │   └── meta.yaml
    └── models/
```