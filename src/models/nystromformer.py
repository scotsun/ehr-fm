import math
import torch
import torch.nn as nn

from transformers.activations import ACT2FN
from transformers.modeling_utils import PreTrainedModel

from . import FMConfig, FMEmbeddings
from src.layers import FFNLUBlock, FFNSwiGLUBlock, ResidualConnection
from src.layers import T2V


class NystromformerSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(
            config, "embedding_size"
        ):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.num_landmarks = config.num_landmarks
        self.seq_len = config.segment_means_seq_len
        self.conv_kernel_size = config.conv_kernel_size

        if config.inv_coeff_init_option:
            self.init_option = config["inv_init_coeff_option"]
        else:
            self.init_option = "original"

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        if self.conv_kernel_size is not None:
            self.conv = nn.Conv2d(
                in_channels=self.num_attention_heads,
                out_channels=self.num_attention_heads,
                kernel_size=(self.conv_kernel_size, 1),
                padding=(self.conv_kernel_size // 2, 0),
                bias=False,
                groups=self.num_attention_heads,
            )

    # Function to approximate Moore-Penrose inverse via the iterative method
    def iterative_inv(self, mat, n_iter=6):
        identity = torch.eye(mat.size(-1), device=mat.device)
        key = mat

        # The entries of key are positive and ||key||_{\infty} = 1 due to softmax
        if self.init_option == "original":
            # This original implementation is more conservative to compute coefficient of Z_0.
            value = 1 / torch.max(torch.sum(key, dim=-2)) * key.transpose(-1, -2)
        else:
            # This is the exact coefficient computation, 1 / ||key||_1, of initialization of Z_0, leading to faster convergence.
            value = (
                1
                / torch.max(torch.sum(key, dim=-2), dim=-1).values[:, :, None, None]
                * key.transpose(-1, -2)
            )

        for _ in range(n_iter):
            key_value = torch.matmul(key, value)
            value = torch.matmul(
                0.25 * value,
                13 * identity
                - torch.matmul(
                    key_value,
                    15 * identity - torch.matmul(key_value, 7 * identity - key_value),
                ),
            )
        return value

    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        batch_size, seq_length, _ = hidden_states.shape
        query_layer = (
            self.query(hidden_states)
            .view(batch_size, -1, self.num_attention_heads, self.attention_head_size)
            .transpose(1, 2)
        )
        key_layer = (
            self.key(hidden_states)
            .view(batch_size, -1, self.num_attention_heads, self.attention_head_size)
            .transpose(1, 2)
        )
        value_layer = (
            self.value(hidden_states)
            .view(batch_size, -1, self.num_attention_heads, self.attention_head_size)
            .transpose(1, 2)
        )

        query_layer = query_layer / math.sqrt(math.sqrt(self.attention_head_size))
        key_layer = key_layer / math.sqrt(math.sqrt(self.attention_head_size))

        if self.num_landmarks == self.seq_len:
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

            if attention_mask is not None:
                # Apply the attention mask is (precomputed for all layers in NystromformerModel forward() function)
                attention_scores = attention_scores + attention_mask

            attention_probs = nn.functional.softmax(attention_scores, dim=-1)
            context_layer = torch.matmul(attention_probs, value_layer)

        else:
            q_landmarks = query_layer.reshape(
                -1,
                self.num_attention_heads,
                self.num_landmarks,
                self.seq_len // self.num_landmarks,
                self.attention_head_size,
            ).mean(dim=-2)
            k_landmarks = key_layer.reshape(
                -1,
                self.num_attention_heads,
                self.num_landmarks,
                self.seq_len // self.num_landmarks,
                self.attention_head_size,
            ).mean(dim=-2)

            kernel_1 = torch.nn.functional.softmax(
                torch.matmul(query_layer, k_landmarks.transpose(-1, -2)), dim=-1
            )
            kernel_2 = torch.nn.functional.softmax(
                torch.matmul(q_landmarks, k_landmarks.transpose(-1, -2)), dim=-1
            )

            attention_scores = torch.matmul(q_landmarks, key_layer.transpose(-1, -2))

            if attention_mask is not None:
                # Apply the attention mask is (precomputed for all layers in NystromformerModel forward() function)
                attention_scores = attention_scores + attention_mask

            kernel_3 = nn.functional.softmax(attention_scores, dim=-1)
            attention_probs = torch.matmul(kernel_1, self.iterative_inv(kernel_2))
            new_value_layer = torch.matmul(kernel_3, value_layer)
            context_layer = torch.matmul(attention_probs, new_value_layer)

        if self.conv_kernel_size is not None:
            context_layer += self.conv(value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (
            (context_layer, attention_probs) if output_attentions else (context_layer,)
        )

        return outputs


# Copied from transformers.models.bert.modeling_bert.BertSelfOutput
class NystromformerSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class NystromformerAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = NystromformerSelfAttention(config)
        self.output = NystromformerSelfOutput(config)

    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        self_outputs = self.self(hidden_states, attention_mask, output_attentions)
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[
            1:
        ]  # add attentions if we output them
        return outputs


# Copied from transformers.models.bert.modeling_bert.BertIntermediate with Bert->Nystromformer
class NystromformerIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertOutput with Bert->Nystromformer
class NystromformerOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class NystromformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn_block = NystromformerSelfAttention(config)
        self.self_attn_out_proj = nn.Linear(config.hidden_size, config.hidden_size)

        norm_type = getattr(config, "norm_type", "layer")
        dropout = getattr(config, "hidden_dropout_prob", 0.0)

        ffn_type = getattr(config, "ffn_type", None)
        if ffn_type is None:
            hidden_act = getattr(config, "hidden_act", "gelu")
            ffn_type = hidden_act if isinstance(hidden_act, str) else "gelu"

        if ffn_type == "swiglu":
            self.ffn_block = FFNSwiGLUBlock(
                config.hidden_size,
                config.intermediate_size,
                dropout=0.0,
            )
        elif ffn_type in {"gelu", "relu"}:
            self.ffn_block = FFNLUBlock(
                config.hidden_size,
                config.intermediate_size,
                activation=ffn_type,
                dropout=0.0,
            )
        else:
            raise ValueError(
                f"Unsupported FFN type/activation: {ffn_type}. Expected 'swiglu', 'gelu', or 'relu'."
            )

        self.residual_connection0 = ResidualConnection(
            config.hidden_size, dropout, norm_type
        )
        self.residual_connection1 = ResidualConnection(
            config.hidden_size, dropout, norm_type
        )

    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        attn_probs = None

        def self_attn_sublayer(x):
            nonlocal attn_probs
            self_attn_out = self.self_attn_block(
                x, attention_mask, output_attentions=output_attentions
            )
            if output_attentions:
                attn_probs = self_attn_out[1]
            return self.self_attn_out_proj(self_attn_out[0])

        hidden_states = self.residual_connection0(hidden_states, self_attn_sublayer)
        hidden_states = self.residual_connection1(hidden_states, self.ffn_block)

        return (hidden_states, attn_probs) if output_attentions else (hidden_states,)


class _NystromLayerConfig:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key)


class FMNystromformer(PreTrainedModel):
    config_class = FMConfig
    model_type = "fm-nystromformer"

    def __init__(self, config: FMConfig):
        super().__init__(config)

        seq_len = config.dataset.get("max_seq", config.dataset.get("max_set_size"))
        if seq_len is None:
            raise ValueError(
                "FMNystromformer requires dataset.max_seq (or dataset.max_set_size)."
            )

        num_landmarks = getattr(config, "num_landmarks", seq_len)
        if num_landmarks <= 0 or seq_len % num_landmarks != 0:
            num_landmarks = seq_len

        nystrom_layer_cfg = _NystromLayerConfig(
            hidden_size=config.d_model,
            num_attention_heads=config.n_heads,
            intermediate_size=config.d_ff,
            hidden_dropout_prob=config.dropout,
            attention_probs_dropout_prob=config.dropout,
            norm_type=config.norm_type,
            ffn_type=config.ffn_type,
            hidden_act="gelu",
            num_landmarks=num_landmarks,
            segment_means_seq_len=seq_len,
            conv_kernel_size=getattr(config, "conv_kernel_size", None),
            inv_coeff_init_option=getattr(config, "inv_coeff_init_option", False),
            inv_init_coeff_option=getattr(config, "inv_init_coeff_option", "original"),
        )

        self.embeddings = FMEmbeddings(config)
        self.t2v = T2V(config.d_model, config.t2v_scale)
        self.blocks = nn.ModuleList(
            [NystromformerLayer(nystrom_layer_cfg) for _ in range(config.n_blocks)]
        )

        match config.norm_type:
            case "layer":
                self.last_norm = nn.LayerNorm(config.d_model, bias=False)
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
        return logits, h

    def encode(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, t: torch.Tensor
    ):
        h = self.embeddings(input_ids)
        h = h + self.t2v(t)

        nystrom_attention_mask = None
        if attention_mask is not None:
            mask = attention_mask[:, None, None, :].to(dtype=h.dtype)
            nystrom_attention_mask = (1.0 - mask) * torch.finfo(h.dtype).min

        for block in self.blocks:
            h = block(h, nystrom_attention_mask)[0]
        h = self.last_norm(h)
        return h
