import torch
import torch.nn as nn

from math import sqrt
from torch import cumsum
from transformers.modeling_utils import PreTrainedModel
from transformers.models.longformer.modeling_longformer import LongformerSelfAttention
from transformers import LongformerConfig

from . import FMConfig, FMEmbeddings
from src.layers import T2V, RoPE, FFNSwiGLUBlock, FFNLUBlock, ResidualConnection


class LongformerMHABlock(LongformerSelfAttention):
    """
    Longformer MHA block with RoPE (applied set positions)

    The forward pass code is adapted from
    `LongformerSelfAttention.forward` in HuggingFace Transformers
    """

    def __init__(self, config, layer_id):
        super().__init__(config, layer_id)
        self.h = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.rope_q = RoPE(self.head_dim)
        self.rope_k = RoPE(self.head_dim)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        is_index_masked=None,
        is_index_global_attn=None,
        is_global_attn=None,
        output_attentions=False,
        head_mask=None,
    ):
        """
        Re-implementation of LongformerSelfAttention.forward with RoPE.
        """
        batch_size, seq_len, embed_dim = hidden_states.shape

        # Original LongformerSelfAttention transposes at the beginning: (seq_len, batch_size, embed_dim)
        hidden_states_transposed = hidden_states.transpose(0, 1)

        # 1. Project QKV
        query_vectors = self.query(hidden_states_transposed)
        key_vectors = self.key(hidden_states_transposed)
        value_vectors = self.value(hidden_states_transposed)

        # normalize query
        query_vectors /= sqrt(self.head_dim)

        # Reshape to (seq_len, batch_size, num_heads, head_dim)
        query_vectors = query_vectors.view(
            seq_len, batch_size, self.num_heads, self.head_dim
        ).transpose(0, 1)
        print(query_vectors.shape)
        key_vectors = key_vectors.view(
            seq_len, batch_size, self.num_heads, self.head_dim
        ).transpose(0, 1)

        # RoPE with set position
        set_pos = (~is_index_masked) * cumsum(is_index_global_attn, dim=1)
        query_vectors = self.rope_q(query_vectors, time=set_pos)
        key_vectors = self.rope_k(key_vectors, time=set_pos)

        # continue with original Longformer logic (sparse attention mechanism)
        attn_scores = self._sliding_chunks_query_key_matmul(
            query_vectors, key_vectors, self.one_sided_attn_window_size
        )

        # values to pad for attention probs
        remove_from_windowed_attention_mask = (attention_mask != 0)[:, :, None, None]

        # cast to fp32/fp16 then replace 1's with -inf
        float_mask = remove_from_windowed_attention_mask.type_as(
            query_vectors
        ).masked_fill(
            remove_from_windowed_attention_mask, torch.finfo(query_vectors.dtype).min
        )
        # diagonal mask with zeros everywhere and -inf inplace of padding
        diagonal_mask = self._sliding_chunks_query_key_matmul(
            float_mask.new_ones(size=float_mask.size()),
            float_mask,
            self.one_sided_attn_window_size,
        )

        # pad local attention probs
        attn_scores += diagonal_mask

        assert list(attn_scores.size()) == [
            batch_size,
            seq_len,
            self.num_heads,
            self.one_sided_attn_window_size * 2 + 1,
        ], (
            f"local_attn_probs should be of size ({batch_size}, {seq_len}, {self.num_heads},"
            f" {self.one_sided_attn_window_size * 2 + 1}), but is of size {attn_scores.size()}"
        )

        # compute local attention probs from global attention keys and contact over window dim
        if is_global_attn:
            # compute global attn indices required through out forward fn
            (
                max_num_global_attn_indices,
                is_index_global_attn_nonzero_orig,
                is_local_index_global_attn_nonzero,
                is_local_index_no_global_attn_nonzero,
            ) = self._get_global_attn_indices(is_index_global_attn)

            # calculate global attn probs from global key
            global_key_attn_scores = self._concat_with_global_key_attn_probs(
                query_vectors=query_vectors,
                key_vectors=key_vectors,
                max_num_global_attn_indices=max_num_global_attn_indices,
                is_index_global_attn_nonzero=is_index_global_attn_nonzero_orig,
                is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero,
                is_local_index_no_global_attn_nonzero=is_local_index_no_global_attn_nonzero,
            )
            # concat to local_attn_probs
            # (batch_size, seq_len, num_heads, extra attention count + 2*window+1)
            attn_scores = torch.cat((global_key_attn_scores, attn_scores), dim=-1)

            # free memory
            del global_key_attn_scores

        attn_probs = nn.functional.softmax(
            attn_scores, dim=-1, dtype=torch.float32
        )  # use fp32 for numerical stability

        # softmax sometimes inserts NaN if all positions are masked, replace them with 0
        attn_probs = torch.masked_fill(
            attn_probs, is_index_masked[:, :, None, None], 0.0
        )
        attn_probs = attn_probs.type_as(attn_scores)

        # free memory
        del attn_scores

        # apply dropout
        attn_probs = nn.functional.dropout(
            attn_probs, p=self.dropout, training=self.training
        )

        value_vectors = value_vectors.view(
            seq_len, batch_size, self.num_heads, self.head_dim
        ).transpose(0, 1)

        # compute local attention output with global attention value and add
        if is_global_attn:
            # compute sum of global and local attn
            attn_output = self._compute_attn_output_with_global_indices(
                value_vectors=value_vectors,
                attn_probs=attn_probs,
                max_num_global_attn_indices=max_num_global_attn_indices,
                is_index_global_attn_nonzero=is_index_global_attn_nonzero_orig,
                is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero,
            )
        else:
            # compute local attn only
            attn_output = self._sliding_chunks_matmul_attn_probs_value(
                attn_probs, value_vectors, self.one_sided_attn_window_size
            )

        assert attn_output.size() == (
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim,
        ), "Unexpected size"
        attn_output = (
            attn_output.transpose(0, 1)
            .reshape(seq_len, batch_size, embed_dim)
            .contiguous()
        )

        # compute value for global attention and overwrite to attention output
        # TODO: remove the redundant computation
        if is_global_attn:
            global_attn_output, global_attn_probs = (
                self._compute_global_attn_output_from_hidden(
                    hidden_states=hidden_states_transposed,  # Corrected: pass transposed
                    max_num_global_attn_indices=max_num_global_attn_indices,
                    is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero,
                    is_index_global_attn_nonzero=is_index_global_attn_nonzero_orig,  # Corrected: pass original
                    is_local_index_no_global_attn_nonzero=is_local_index_no_global_attn_nonzero,
                    is_index_masked=is_index_masked,
                    layer_head_mask=head_mask,
                )
            )

            # get only non zero global attn output
            nonzero_global_attn_output = global_attn_output[
                is_local_index_global_attn_nonzero[0],
                :,
                is_local_index_global_attn_nonzero[1],
            ]

            # overwrite values with global attention
            attn_output[is_index_global_attn_nonzero_orig[::-1]] = (
                nonzero_global_attn_output.view(
                    len(is_local_index_global_attn_nonzero[0]), -1
                )
            )
            # The attention weights for tokens with global attention are
            # just filler values, they were never used to compute the output.
            # Fill with 0 now, the correct values are in 'global_attn_probs'.
            attn_probs[is_index_global_attn_nonzero_orig] = 0

        outputs = (attn_output.transpose(0, 1),)

        if output_attentions:
            outputs += (attn_probs,)

        return (
            outputs + (global_attn_probs,)
            if (is_global_attn and output_attentions)
            else outputs
        )


class LongformerBlock(nn.Module):
    def __init__(self, config: LongformerConfig, layer_id: int):
        super().__init__()
        self.longformer_self_attn_block = LongformerMHABlock(config, layer_id)
        self.ffn_block = (
            FFNSwiGLUBlock(config.hidden_size, config.intermediate_size)
            if config.ffn_type == "swiglu"
            else FFNLUBlock(
                config.hidden_size, config.intermediate_size, config.ffn_type
            )
        )
        self.residual_connection0 = ResidualConnection(
            config.hidden_size, config.hidden_dropout_prob, config.norm_type
        )
        self.residual_connection1 = ResidualConnection(
            config.hidden_size, config.hidden_dropout_prob, config.norm_type
        )

    def _merge_to_attention_mask(
        self, attention_mask: torch.Tensor, global_attention_mask: torch.Tensor
    ):
        # longformer self attention expects attention mask to have 0 (no attn), 1 (local attn), 2 (global attn)
        # (global_attention_mask + 1) => 1 for local attention, 2 for global attention
        # => final attention_mask => 0 for no attention, 1 for local attention 2 for global attention
        if attention_mask is not None:
            attention_mask = attention_mask * (global_attention_mask + 1)
        else:
            # simply use `global_attention_mask` as `attention_mask`
            # if no `attention_mask` is given
            attention_mask = global_attention_mask + 1
        return attention_mask

    def forward(self, x, attention_mask, global_attention_mask):
        attention_mask = self._merge_to_attention_mask(
            attention_mask, global_attention_mask
        )
        x = self.residual_connection0(
            x,
            self.longformer_self_attn_block(
                hidden_states=x,
                attention_mask=attention_mask,
                is_index_global_attn=attention_mask == 2,
                is_index_masked=attention_mask == 0,
                is_global_attn=True,
            ),
        )
        x = self.residual_connection1(x, self.ffn_block(x))
        return x


class FMLongformer(PreTrainedModel):
    config_class = FMConfig
    model_type = "fm-longformer"

    def __init__(self, config: FMConfig):
        super().__init__(config)
        longformer_cfg = LongformerConfig(  # longformer cfg interface
            attention_window=[config.dataset["max_set_size"]] * config.n_blocks,
            hidden_size=config.d_model,
            num_attention_heads=config.n_heads,
            num_hidden_layers=config.n_blocks,
            intermediate_size=config.d_ff,
            hidden_dropout_prob=config.dropout,
            norm_type=config.norm_type,
            ffn_type=config.ffn_type,
        )
        self.embeddings = FMEmbeddings(config)
        self.t2v = T2V(config.d_model, config.t2v_scale)
        self.blocks = nn.ModuleList(
            [
                LongformerBlock(longformer_cfg, layer_id)
                for layer_id in range(longformer_cfg.num_hidden_layers)
            ]
        )

        match config.norm_type:
            case "layer":
                self.last_norm = nn.LayerNorm(config.d_model)
            case "rms":
                self.last_norm = nn.RMSNorm(config.d_model)
            case _:
                raise ValueError(f"{config.norm_type} not implemented")

        self.lm_head = nn.Linear(config.d_model, config.vocab_size)
        if config.weight_tying:
            self.lm_head.weight = self.embeddings.embeddings.weight

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, t: torch.Tensor
    ):
        h = self.encode(input_ids, attention_mask, t)
        logits = self.lm_head(h)
        # (batch, max_seq, max_set_size, d_model)
        return logits, h

    def encode(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, t: torch.Tensor
    ):
        global_attention_mask = input_ids == 2
        h = self.embeddings(input_ids)
        h = self.t2v(t)
        for block in self.blocks:
            h = block(h, attention_mask, global_attention_mask)
        h = self.last_norm(h)
        return h
