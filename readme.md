# todo:
- [ ] add dataset util
- [ ] add rope
- [ ] allow attention retrieval from transformer layer




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