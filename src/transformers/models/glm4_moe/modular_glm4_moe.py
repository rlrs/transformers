# Copyright 2025 The ZhipuAI Inc. team and HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch GLM-4-MOE model."""

from dataclasses import dataclass

import torch
from torch import nn

from ...cache_utils import Cache
from ...configuration_utils import PreTrainedConfig
from ...masking_utils import create_causal_mask
from ...modeling_outputs import BaseModelOutputWithPast, ModelOutput
from ...modeling_rope_utils import RopeParameters
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from ..cohere.modeling_cohere import CohereAttention
from ..deepseek_v3.modeling_deepseek_v3 import (
    DeepseekV3DecoderLayer,
    DeepseekV3ForCausalLM,
    DeepseekV3MLP,
    DeepseekV3Model,
    DeepseekV3PreTrainedModel,
    DeepseekV3RMSNorm,
    DeepseekV3TopkRouter,
)
from ..glm.modeling_glm import GlmRotaryEmbedding
from ..gpt_neox.modeling_gpt_neox import apply_rotary_pos_emb  # noqa


logger = logging.get_logger(__name__)


class Glm4MoeConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Glm4MoeModel`]. It is used to instantiate a
    Glm4Moe model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of [THUDM/GLM-4-100B-A10B](https://huggingface.co/THUDM/GLM-4-100B-A10B).

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 151552):
            Vocabulary size of the Glm4Moe model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`Glm4MoeModel`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 10944):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 46):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 96):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*, defaults to 8):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details, check out [this
            paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to `32`.

        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 131072):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the model's input and output word embeddings should be tied.
        rope_parameters (`RopeParameters`, *optional*):
            Dictionary containing the configuration parameters for the RoPE embeddings. The dictionary should contain
            a value for `rope_theta` and optionally parameters used for scaling in case you want to use RoPE
            with longer `max_position_embeddings`.
        attention_bias (`bool`, defaults to `False`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        moe_intermediate_size (`int`, *optional*, defaults to 1408):
            Intermediate size of the routed expert.
        num_experts_per_tok (`int`, *optional*, defaults to 8):
            number of experts per token.
        n_shared_experts (`int`, *optional*, defaults to 1):
            Number of shared experts.
        n_routed_experts (`int`, *optional*, defaults to 128):
            Number of routed experts.
        routed_scaling_factor (`float`, *optional*, defaults to 1.0):
            Scaling factor or routed experts.
        n_group (`int`, *optional*, defaults to 1):
            Number of groups for routed experts.
        topk_group (`int`, *optional*, defaults to 1):
            Number of selected groups for each token(for each token, ensuring the selected experts is only within `topk_group` groups).
        first_k_dense_replace (`int`, *optional*, defaults to 1):
            Number of dense layers in shallow layers(embed->dense->dense->...->dense->moe->moe...->lm_head).
                                                            \--k dense layers--/
        num_nextn_predict_layers (`int`, *optional*, defaults to 1):
            Number of MTP layers stacked after the base decoder.
        mtp_lambda_weight (`float`, *optional*, defaults to 0.3):
            Weight used for MTP loss during training.
        norm_topk_prob (`bool`, *optional*, defaults to `True`):
            Whether to normalize the topk probabilities.
        use_qk_norm (`bool`, *optional*, defaults to `False`):
            Whether to use query-key normalization in the attention
        bos_token_id (`int`, *optional*):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*):
            End of stream token id.
        pad_token_id (`int`, *optional*):
            Padding token id.

    ```python
    >>> from transformers import Glm4MoeModel, Glm4MoeConfig

    >>> # Initializing a Glm4Moe style configuration
    >>> configuration = Glm4MoeConfig()

    >>> # Initializing a model from the GLM-4-MOE-100B-A10B style configuration
    >>> model = Glm4MoeModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "glm4_moe"
    keys_to_ignore_at_inference = ["past_key_values"]

    # Default tensor parallel plan for base model `Glm4Moe`
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.experts.gate_up_proj": "rowwise",
        "layers.*.mlp.experts.down_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }
    attribute_map = {
        "num_local_experts": "n_routed_experts",
    }

    def __init__(
        self,
        vocab_size: int | None = 151552,
        hidden_size: int | None = 4096,
        intermediate_size: int | None = 10944,
        num_hidden_layers: int | None = 46,
        num_attention_heads: int | None = 96,
        num_key_value_heads: int | None = 8,
        hidden_act: str | None = "silu",
        max_position_embeddings: int | None = 131072,
        initializer_range: float | None = 0.02,
        rms_norm_eps: int | None = 1e-5,
        use_cache: bool | None = True,
        tie_word_embeddings: bool | None = False,
        rope_parameters: RopeParameters | dict[str, RopeParameters] | None = None,
        attention_bias: bool | None = False,
        attention_dropout: float | None = 0.0,
        moe_intermediate_size: int | None = 1408,
        num_experts_per_tok: int | None = 8,
        n_shared_experts: int | None = 1,
        n_routed_experts: int | None = 128,
        routed_scaling_factor: float | None = 1.0,
        n_group: int | None = 1,
        topk_group: int | None = 1,
        first_k_dense_replace: int | None = 1,
        num_nextn_predict_layers: int | None = 1,
        mtp_lambda_weight: float | None = 0.3,
        norm_topk_prob: bool | None = True,
        use_qk_norm: bool | None = False,
        bos_token_id: int | None = None,
        eos_token_id: int | None = None,
        pad_token_id: int | None = None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.rope_parameters = rope_parameters
        kwargs.setdefault("partial_rotary_factor", 0.5)  # assign default for BC

        # MoE arguments
        self.moe_intermediate_size = moe_intermediate_size
        self.num_experts_per_tok = num_experts_per_tok
        self.n_group = n_group
        self.topk_group = topk_group
        self.n_shared_experts = n_shared_experts
        self.n_routed_experts = n_routed_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.first_k_dense_replace = first_k_dense_replace
        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.mtp_lambda_weight = mtp_lambda_weight
        self.norm_topk_prob = norm_topk_prob
        self.use_qk_norm = use_qk_norm
        self.tie_word_embeddings = tie_word_embeddings
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

        super().__init__(**kwargs)


class Glm4MoeRotaryEmbedding(GlmRotaryEmbedding):
    pass


class Glm4MoeAttention(CohereAttention):
    def __init__(self, config: Glm4MoeConfig, layer_idx: int | None = None):
        nn.Module.__init__(self)
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.rope_parameters = config.rope_parameters
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        self.use_qk_norm = config.use_qk_norm
        if self.use_qk_norm:
            self.q_norm = Glm4MoeRMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = Glm4MoeRMSNorm(self.head_dim, eps=config.rms_norm_eps)


class Glm4MoeMLP(DeepseekV3MLP):
    pass


class Glm4MoeTopkRouter(DeepseekV3TopkRouter):
    def __init__(self, config: Glm4MoeConfig):
        nn.Module.__init__(self)
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob

        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, config.hidden_size)))
        self.register_buffer("e_score_correction_bias", torch.zeros((self.n_routed_experts), dtype=torch.float32))


class Glm4MoeRMSNorm(DeepseekV3RMSNorm):
    pass


class Glm4MoeDecoderLayer(DeepseekV3DecoderLayer):
    pass


class Glm4MoeSharedHead(nn.Module):
    def __init__(self, config: Glm4MoeConfig):
        super().__init__()
        self.norm = Glm4MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.head(self.norm(hidden_states))


class Glm4MoeMultiTokenPredictorLayer(Glm4MoeDecoderLayer):
    def __init__(self, config: Glm4MoeConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.enorm = Glm4MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = Glm4MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)
        self.shared_head = Glm4MoeSharedHead(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.LongTensor,
        embed_tokens: nn.Embedding,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_embeds = embed_tokens(input_ids)
        if hidden_states.shape[:2] != input_embeds.shape[:2]:
            raise ValueError("MTP hidden states and shifted inputs must have matching sequence shapes.")

        hidden_states = self.eh_proj(torch.cat((self.enorm(input_embeds), self.hnorm(hidden_states)), dim=-1))
        hidden_states = super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        logits = self.shared_head(hidden_states)
        return hidden_states, logits


class Glm4MoePreTrainedModel(DeepseekV3PreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"model\.layers\.\d+\.shared_head\.head\..*"]


class Glm4MoeModel(DeepseekV3Model):
    def __init__(self, config: Glm4MoeConfig):
        super().__init__(config)
        self.layers.extend(
            [
                Glm4MoeMultiTokenPredictorLayer(config, layer_idx)
                for layer_idx in range(
                    config.num_hidden_layers, config.num_hidden_layers + config.num_nextn_predict_layers
                )
            ]
        )

    def forward_mtp(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, ...]:
        if self.config.num_nextn_predict_layers == 0 or hidden_states.shape[1] <= 1:
            return ()

        hidden_states = hidden_states[:, :-1]
        mtp_logits = []
        num_nextn_predict_tokens = min(input_ids.shape[1] - 1, self.config.num_nextn_predict_layers)
        for mtp_idx in range(num_nextn_predict_tokens):
            shifted_input_ids = input_ids[:, mtp_idx + 1 :]
            if shifted_input_ids.shape[1] == 0:
                break

            if position_ids is None:
                shifted_position_ids = torch.arange(
                    shifted_input_ids.shape[1], device=shifted_input_ids.device, dtype=torch.long
                )
                shifted_position_ids = shifted_position_ids.unsqueeze(0).expand(shifted_input_ids.shape[0], -1)
            else:
                shifted_position_ids = position_ids[:, mtp_idx + 1 :]

            if cache_position is None:
                shifted_cache_position = torch.arange(
                    shifted_input_ids.shape[1], device=shifted_input_ids.device, dtype=torch.long
                )
            else:
                shifted_cache_position = cache_position[mtp_idx + 1 :]

            shifted_attention_mask = attention_mask[:, mtp_idx + 1 :] if attention_mask is not None else None
            shifted_inputs_embeds = self.embed_tokens(shifted_input_ids)
            shifted_causal_mask = create_causal_mask(
                config=self.config,
                inputs_embeds=shifted_inputs_embeds,
                attention_mask=shifted_attention_mask,
                cache_position=shifted_cache_position,
                past_key_values=None,
                position_ids=shifted_position_ids,
            )
            shifted_position_embeddings = self.rotary_emb(hidden_states, shifted_position_ids)

            mtp_layer = self.layers[self.config.num_hidden_layers + mtp_idx]
            hidden_states, logits = mtp_layer(
                hidden_states=hidden_states,
                input_ids=shifted_input_ids,
                embed_tokens=self.embed_tokens,
                attention_mask=shifted_causal_mask,
                position_ids=shifted_position_ids,
                use_cache=False,
                cache_position=shifted_cache_position,
                position_embeddings=shifted_position_embeddings,
                **kwargs,
            )
            mtp_logits.append(logits)
            hidden_states = hidden_states[:, :-1]
            if hidden_states.shape[1] <= 1:
                break

        return tuple(mtp_logits)


@dataclass
@auto_docstring(
    custom_intro="""
    Base class for GLM-4-MoE causal language model (or autoregressive) outputs with optional MTP logits.
    """
)
class Glm4MoeCausalLMOutputWithPast(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Language modeling loss (for next-token prediction).
    logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
        Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

        Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
        `past_key_values` input) to speed up sequential decoding.
    mtp_logits (`tuple(torch.FloatTensor)`, *optional*):
        Prediction scores from MTP layers. Each tensor has shape
        `(batch_size, sequence_length - i - 1, config.vocab_size)` for the i-th MTP layer.
    """

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    past_key_values: Cache | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None
    mtp_logits: tuple[torch.FloatTensor, ...] | None = None


class Glm4MoeForCausalLM(DeepseekV3ForCausalLM):
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        output_mtp_logits: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Glm4MoeCausalLMOutputWithPast:
        r"""
        output_mtp_logits (`bool`, *optional*, defaults to `False`):
            Whether to return logits produced by MTP layers.
        """
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        mtp_logits = None
        should_compute_mtp = (
            self.config.num_nextn_predict_layers > 0
            and input_ids is not None
            and hidden_states.shape[1] > 1
            and past_key_values is None
            and (output_mtp_logits or labels is not None)
        )
        if should_compute_mtp:
            mtp_logits = self.model.forward_mtp(
                hidden_states=hidden_states,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                cache_position=cache_position,
                **kwargs,
            )

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
            if mtp_logits is not None and len(mtp_logits) > 0:
                mtp_losses = []
                for mtp_idx, mtp_layer_logits in enumerate(mtp_logits):
                    shifted_labels = labels[:, mtp_idx + 1 :]
                    if shifted_labels.shape[1] <= 1:
                        continue
                    mtp_losses.append(
                        self.loss_function(
                            logits=mtp_layer_logits, labels=shifted_labels, vocab_size=self.config.vocab_size, **kwargs
                        )
                    )
                if len(mtp_losses) > 0:
                    mtp_loss = torch.stack(mtp_losses).mean()
                    loss = loss + self.config.mtp_lambda_weight * mtp_loss

        if not output_mtp_logits:
            mtp_logits = None

        return Glm4MoeCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            mtp_logits=mtp_logits,
        )


__all__ = [
    "Glm4MoeConfig",
    "Glm4MoeCausalLMOutputWithPast",
    "Glm4MoePreTrainedModel",
    "Glm4MoeModel",
    "Glm4MoeForCausalLM",
]
