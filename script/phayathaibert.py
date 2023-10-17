# coding=utf-8
# Copyright 2019 Inria, Facebook AI Research and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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

"""
        ## PyTorch PhayaThaiBERT model ##
These models aren't meant to be used by end users.
They are developed so that the added vocab's embeddings
can be trained separately from the original vocab's embeddings.
See the PhayaThaiBERTEmbeddings class for how this is implemented.
"""

import math, torch
from torch.utils.checkpoint import checkpoint
from torch.nn.functional import softmax, pad
from torch.nn import (
    Module,
    ModuleList,
    Embedding,
    LayerNorm,
    Dropout,
    Linear,
    Tanh,
    Parameter,
    CrossEntropyLoss
)
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from transformers.models.camembert.modeling_camembert import CamembertForMaskedLM
from transformers.models.camembert.configuration_camembert import CamembertConfig
from transformers.activations import ACT2FN, gelu
from transformers.utils.logging import get_logger
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    MaskedLMOutput,
)
from transformers.pytorch_utils import (
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer
)

logger = get_logger(__name__)

class PhayaThaiBERTConfig(PretrainedConfig):

    model_type = "phayathaibert"

    def __init__(
        self,
        old_vocab_size=25005,
        new_vocab_size=224257,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        position_embedding_type="absolute",
        use_cache=True,
        classifier_dropout=None,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs
        )

        self.old_vocab_size = old_vocab_size
        self.new_vocab_size = new_vocab_size
        self.vocab_size = old_vocab_size + new_vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout

    @classmethod
    def from_camembert_config(
        cls,
        camembert_config: CamembertConfig,
        num_total_vocab: int
    ):
        return cls(
            old_vocab_size=camembert_config.vocab_size,
            new_vocab_size=num_total_vocab - camembert_config.vocab_size,
            hidden_size=camembert_config.hidden_size,
            num_hidden_layers=camembert_config.num_hidden_layers,
            num_attention_heads=camembert_config.num_attention_heads,
            intermediate_size=camembert_config.intermediate_size,
            hidden_act=camembert_config.hidden_act,
            hidden_dropout_prob=camembert_config.hidden_dropout_prob,
            attention_probs_dropout_prob=camembert_config.attention_probs_dropout_prob,
            max_position_embeddings=camembert_config.max_position_embeddings,
            type_vocab_size=camembert_config.type_vocab_size,
            initializer_range=camembert_config.initializer_range,
            layer_norm_eps=camembert_config.layer_norm_eps,
            pad_token_id=camembert_config.pad_token_id,
            bos_token_id=camembert_config.bos_token_id,
            eos_token_id=camembert_config.eos_token_id,
            position_embedding_type=camembert_config.position_embedding_type,
            use_cache=camembert_config.use_cache,
            classifier_dropout=camembert_config.classifier_dropout,
        )

    def to_camembert_config(self):
        return CamembertConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
            initializer_range=self.initializer_range,
            layer_norm_eps=self.layer_norm_eps,
            pad_token_id=self.pad_token_id,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            position_embedding_type=self.position_embedding_type,
            use_cache=self.use_cache,
            classifier_dropout=self.classifier_dropout,
        )

# Copied from transformers.models.camembert.modeling_camembert.CamembertEmbeddings with Camembert->PhayaThaiBERT
class PhayaThaiBERTEmbeddings(Module):

    def __init__(self, config: PhayaThaiBERTConfig):
        super().__init__()
        self.num_old_embeddings = config.old_vocab_size
        self.embedding_dim = config.hidden_size
        self.old_word_embeddings = Embedding(config.old_vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.new_word_embeddings = Embedding(config.new_vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )
        # End copy
        self.padding_idx = config.pad_token_id
        self.position_embeddings = Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        past_key_values_length=0
    ):
        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = self.create_position_ids_from_input_ids(input_ids, past_key_values_length)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            # BEGIN NEW LOGIC
            id_mask = (input_ids < self.num_old_embeddings)
            old_inputs_embeds = self.old_word_embeddings(id_mask * input_ids)
            new_inputs_embeds = self.new_word_embeddings(~id_mask * (input_ids - self.num_old_embeddings))
            embed_mask = id_mask.unsqueeze(2).expand(-1, -1, self.embedding_dim)
            inputs_embeds = (embed_mask * old_inputs_embeds) + (~embed_mask * new_inputs_embeds)
            # END NEW LOGIC
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.

        Args:
            inputs_embeds: torch.Tensor

        Returns: torch.Tensor
        """
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        return position_ids.unsqueeze(0).expand(input_shape)

    # Copied from transformers.models.camembert.modeling_camembert.create_position_ids_from_input_ids
    def create_position_ids_from_input_ids(self, input_ids, past_key_values_length=0):
        """
        Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
        are ignored. This is modified from fairseq's `utils.make_positions`.

        Args:
            x: torch.Tensor x:

        Returns: torch.Tensor
        """
        # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
        mask = input_ids.ne(self.padding_idx).int()
        incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
        return incremental_indices.long() + self.padding_idx


# Copied from transformers.models.camembert.modeling_camembert.CamembertSelfAttention with Camembert->PhayaThaiBERT
class PhayaThaiBERTSelfAttention(Module):
    def __init__(self, config: PhayaThaiBERTConfig, position_embedding_type=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.dropout = Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, *args, **kwargs):
        if torch.is_autocast_enabled():
            # This temporarily disable mixed precision for numerical stability
            with torch.cuda.amp.autocast(enabled=False):
                return self._forward(*args, **kwargs)
        else:
            return self._forward(*args, **kwargs)

    def _forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None = None,
        head_mask: torch.FloatTensor | None = None,
        encoder_hidden_states: torch.FloatTensor | None = None,
        encoder_attention_mask: torch.FloatTensor | None = None,
        past_key_value: tuple[tuple[torch.FloatTensor]] | None = None,
        output_attentions: bool | None = False,
    ) -> tuple[torch.Tensor]:
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        use_cache = past_key_value is not None
        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            if use_cache:
                position_ids_l = torch.tensor(key_length - 1, dtype=torch.long, device=hidden_states.device).view(
                    -1, 1
                )
            else:
                position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in PhayaThaiBERTModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


# Copied from transformers.models.camembert.modeling_camembert.CamembertSelfOutput with Camembert->PhayaThaiBERT
class PhayaThaiBERTSelfOutput(Module):
    def __init__(self, config: PhayaThaiBERTConfig):
        super().__init__()
        self.dense = Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# Copied from transformers.models.camembert.modeling_camembert.CamembertAttention with Camembert->PhayaThaiBERT
class PhayaThaiBERTAttention(Module):
    def __init__(self, config: PhayaThaiBERTConfig, position_embedding_type=None):
        super().__init__()
        self.self = PhayaThaiBERTSelfAttention(config, position_embedding_type=position_embedding_type)
        self.output = PhayaThaiBERTSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None = None,
        head_mask: torch.FloatTensor | None = None,
        encoder_hidden_states: torch.FloatTensor | None = None,
        encoder_attention_mask: torch.FloatTensor | None = None,
        past_key_value: tuple[tuple[torch.FloatTensor]] | None = None,
        output_attentions: bool | None = False,
    ) -> tuple[torch.Tensor]:
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


# Copied from transformers.models.bert.modeling_bert.BertIntermediate with Bert->Roberta->Camembert->PhayaThaiBERT
class PhayaThaiBERTIntermediate(Module):
    def __init__(self, config: PhayaThaiBERTConfig):
        super().__init__()
        self.dense = Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertOutput with Bert->Roberta->Camembert->PhayaThaiBERT
class PhayaThaiBERTOutput(Module):
    def __init__(self, config: PhayaThaiBERTConfig):
        super().__init__()
        self.dense = Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# Copied from transformers.models.camembert.modeling_camembert.CamembertLayer with Camembert->PhayaThaiBERT
class PhayaThaiBERTLayer(Module):
    def __init__(self, config: PhayaThaiBERTConfig):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = PhayaThaiBERTAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = PhayaThaiBERTAttention(config, position_embedding_type="absolute")
        self.intermediate = PhayaThaiBERTIntermediate(config)
        self.output = PhayaThaiBERTOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None = None,
        head_mask: torch.FloatTensor | None = None,
        encoder_hidden_states: torch.FloatTensor | None = None,
        encoder_attention_mask: torch.FloatTensor | None = None,
        past_key_value: tuple[tuple[torch.FloatTensor]] | None = None,
        output_attentions: bool | None = False,
    ) -> tuple[torch.Tensor]:
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


# Copied from transformers.models.camembert.modeling_camembert.CamembertEncoder with Camembert->PhayaThaiBERT
class PhayaThaiBERTEncoder(Module):
    def __init__(self, config: PhayaThaiBERTConfig):
        super().__init__()
        self.config = config
        self.layer = ModuleList([PhayaThaiBERTLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None = None,
        head_mask: torch.FloatTensor | None = None,
        encoder_hidden_states: torch.FloatTensor | None = None,
        encoder_attention_mask: torch.FloatTensor | None = None,
        past_key_values: tuple[tuple[torch.FloatTensor]] | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = False,
        output_hidden_states: bool | None = False,
        return_dict: bool | None = True,
    ) -> tuple[torch.Tensor] | BaseModelOutputWithPastAndCrossAttentions:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


# Copied from transformers.models.bert.modeling_bert.BertPooler
class PhayaThaiBERTPooler(Module):
    def __init__(self, config: PhayaThaiBERTConfig):
        super().__init__()
        self.dense = Linear(config.hidden_size, config.hidden_size)
        self.activation = Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class PhayaThaiBERTPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = PhayaThaiBERTConfig
    base_model_prefix = "roberta"
    supports_gradient_checkpointing = True

    # Copied from transformers.models.bert.modeling_bert.BertPreTrainedModel._init_weights
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, PhayaThaiBERTEncoder):
            module.gradient_checkpointing = value

    def update_keys_to_ignore(self, config, del_keys_to_ignore):
        """Remove some keys from ignore list"""
        if not config.tie_word_embeddings:
            # must make a new list, or the class variable gets modified!
            self._keys_to_ignore_on_save = [k for k in self._keys_to_ignore_on_save if k not in del_keys_to_ignore]
            self._keys_to_ignore_on_load_missing = [
                k for k in self._keys_to_ignore_on_load_missing if k not in del_keys_to_ignore
            ]

# Copied from transformers.models.camembert.modeling_camembert.CamembertLMHead with Camembert->PhayaThaiBERT
class PhayaThaiBERTLMHead(Module):
    """PhayaThaiBERT Head for masked language modeling."""

    def __init__(self, config: PhayaThaiBERTConfig):
        super().__init__()
        self.dense = Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.old_decoder = Linear(config.hidden_size, config.old_vocab_size)
        self.old_bias = Parameter(torch.zeros(config.old_vocab_size))
        self.old_decoder.bias = self.old_bias

        self.new_decoder = Linear(config.hidden_size, config.new_vocab_size)
        self.new_bias = Parameter(torch.zeros(config.new_vocab_size))
        self.new_decoder.bias = self.new_bias

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        old_x = self.old_decoder(x)
        new_x = self.new_decoder(x)
        x = torch.concat((old_x, new_x), dim=2)

        return x

    def _tie_weights(self):
        # To tie those two weights if they get disconnected (on TPU or when the bias is resized)
        # For accelerate compatibility and to not break backward compatibility
        if self.old_decoder.bias.device.type == "meta":
            self.old_decoder.bias = self.old_bias
        else:
            self.old_bias = self.old_decoder.bias

        if self.new_decoder.bias.device.type == "meta":
            self.new_decoder.bias = self.new_bias
        else:
            self.new_bias = self.new_decoder.bias

class PhayaThaiBERTModel(PhayaThaiBERTPreTrainedModel):
    """
    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in *Attention is
    all you need*_ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz
    Kaiser and Illia Polosukhin.

    To behave as a decoder the model needs to be initialized with the `is_decoder` argument of the configuration set to
    `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.

    .. _*Attention is all you need*: https://arxiv.org/abs/1706.03762
    """

    _keys_to_ignore_on_load_missing = [r"position_ids"]
    _no_split_modules = []

    # Copied from transformers.models.bert.modeling_bert.BertModel.__init__ with Bert->Roberta->Camembert->PhayaThaiBERT
    def __init__(self, config: PhayaThaiBERTConfig, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = PhayaThaiBERTEmbeddings(config)
        self.encoder = PhayaThaiBERTEncoder(config)

        self.pooler = PhayaThaiBERTPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.old_word_embeddings, self.embeddings.new_word_embeddings

    def set_input_embeddings(self, old, new):
        self.embeddings.old_word_embeddings = old
        self.embeddings.new_word_embeddings = new

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    # Copied from transformers.models.bert.modeling_bert.BertModel.forward
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> tuple[torch.Tensor] | BaseModelOutputWithPoolingAndCrossAttentions:
        """
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )

# Copied from transformers.models.camembert.modeling_camembert.CamembertForMaskedLM with Camembert->PhayaThaiBERT
class PhayaThaiBERTForMaskedLM(PhayaThaiBERTPreTrainedModel):

    _keys_to_ignore_on_save = [
        r"lm_head.old_decoder.weight",
        r"lm_head.old_decoder.bias",
        r"lm_head.new_decoder.weight",
        r"lm_head.new_decoder.bias"
    ]
    _keys_to_ignore_on_load_missing = [
        r"position_ids",
        r"lm_head.old_decoder.weight",
        r"lm_head.old_decoder.bias",
        r"lm_head.new_decoder.weight",
        r"lm_head.new_decoder.bias"
    ]
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config: PhayaThaiBERTConfig):
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `PhayaThaiBERTForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.roberta = PhayaThaiBERTModel(config, add_pooling_layer=False)
        self.lm_head = PhayaThaiBERTLMHead(config)

        # The LM head weights require special treatment only when they are tied with the word embeddings
        self.update_keys_to_ignore(
            config,
            ["lm_head.old_decoder.weight", "lm_head.new_decoder.weight"]
        )

        # Initialize weights and apply final processing
        self.post_init()

    @classmethod
    def from_camembert_for_masked_lm(
        cls,
        camembert_for_masked_lm: CamembertForMaskedLM,
        num_total_vocab: int
    ):
        config = PhayaThaiBERTConfig.from_camembert_config(
            camembert_for_masked_lm.config,
            num_total_vocab
        )
        model = cls(config)
        model.load_state_dict(camembert_for_masked_lm.state_dict(), strict=False)
        model.roberta.embeddings.old_word_embeddings.weight.data.copy_(
            camembert_for_masked_lm.roberta.embeddings.word_embeddings.weight.data
        )
        model.lm_head.old_decoder.bias.data.copy_(
            camembert_for_masked_lm.lm_head.decoder.bias.data
        )
        model.tie_weights()
        return model

    def to_camembert_for_masked_lm(self):
        camembert_config = self.config.to_camembert_config()
        camembert_for_masked_lm = CamembertForMaskedLM(camembert_config)
        camembert_for_masked_lm.load_state_dict(self.state_dict(), strict=False)
        camembert_for_masked_lm.roberta.embeddings.word_embeddings.weight.data.copy_(
            torch.cat([
                self.roberta.embeddings.old_word_embeddings.weight.data,
                self.roberta.embeddings.new_word_embeddings.weight.data
            ])
        )
        camembert_for_masked_lm.lm_head.decoder.bias.data.copy_(
            torch.cat([
                self.lm_head.old_decoder.bias.data,
                self.lm_head.new_decoder.bias.data
            ])
        )
        camembert_for_masked_lm.tie_weights()
        return camembert_for_masked_lm

    def get_output_embeddings(self):
        return self.lm_head.old_decoder, self.lm_head.new_decoder

    def set_output_embeddings(self, old, new):
        self.lm_head.old_decoder = old
        self.lm_head.new_decoder = new

    def _tie_or_clone_weights(
        self,
        output_embeddings: tuple[Linear, Linear],
        input_embeddings: tuple[Embedding, Embedding]
    ):
        """Tie or clone module weights depending of whether we are using TorchScript or not"""
        old_output_embeddings, new_output_embeddings = output_embeddings
        old_input_embeddings, new_input_embeddings = input_embeddings

        if self.config.torchscript:
            old_output_embeddings.weight = Parameter(old_input_embeddings.weight.clone())
            new_output_embeddings.weight = Parameter(new_input_embeddings.weight.clone())
        else:
            old_output_embeddings.weight = old_input_embeddings.weight
            new_output_embeddings.weight = new_input_embeddings.weight

        old_output_embeddings.bias.data = pad(
            old_output_embeddings.bias.data,
            (
                0,
                old_output_embeddings.weight.shape[0] - old_output_embeddings.bias.shape[0],
            ),
            "constant",
            0,
        )
        new_output_embeddings.bias.data = pad(
            new_output_embeddings.bias.data,
            (
                0,
                new_output_embeddings.weight.shape[0] - new_output_embeddings.bias.shape[0],
            ),
            "constant",
            0,
        )

        old_output_embeddings.out_features = old_input_embeddings.num_embeddings
        new_output_embeddings.out_features = new_input_embeddings.num_embeddings

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.FloatTensor | None = None,
        token_type_ids: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        head_mask: torch.FloatTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        encoder_hidden_states: torch.FloatTensor | None = None,
        encoder_attention_mask: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> tuple[torch.Tensor] | MaskedLMOutput:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        kwargs (`Dict[str, any]`, optional, defaults to *{}*):
            Used to hide legacy arguments that have been deprecated.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(prediction_scores.device)
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
