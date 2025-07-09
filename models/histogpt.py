""" 
HistoGPT Vision Language Model
© Manuel Tran / Helmholtz Munich
"""

import torch
import torch.nn as nn

from flamingo_pytorch import GatedCrossAttentionBlock
from torch.utils.checkpoint import checkpoint

from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def setup_fine_tuning(model, config):
    """
    Configure model parameters for fine-tuning based on config.
    """
    # Freeze/unfreeze components based on config
    if hasattr(model, 'aggregator'):
        requires_grad(model.aggregator, config.train_aggregator)
    
    if hasattr(model, 'projection'):
        requires_grad(model.projection, config.train_projection)
    
    # Handle BioGPT layers
    if hasattr(model, 'histogpt'):
        # Cross-attention layers
        for layer_pair in model.histogpt.layers:
            xattn_layer, decoder_layer = layer_pair
            requires_grad(xattn_layer, config.train_cross_attention)
            requires_grad(decoder_layer, not config.freeze_language_model)
        
        # Embeddings
        requires_grad(model.histogpt.embed_tokens, not config.freeze_language_model)
        requires_grad(model.histogpt.embed_positions, config.train_positional_embedding)
        requires_grad(model.histogpt.layer_norm, not config.freeze_language_model)
    
    # Output projection
    if hasattr(model, 'output_projection'):
        requires_grad(model.output_projection, not config.freeze_language_model)
    
    return model


class HistoGPTModel(nn.Module):
    """
    The HistoGPT model for generating pathology reports.
    """
    def __init__(self, aggregator: nn.Module, biogpt: nn.Module, checkpoint: bool):
        super().__init__()
        self.biogpt_config = biogpt.config
        self.dropout = biogpt.dropout
        self.embed_dim = biogpt.embed_dim
        self.embed_scale = biogpt.embed_scale
        self.layerdrop = biogpt.layerdrop
        self.padding_idx = biogpt.padding_idx

        self.aggregator = aggregator
        self.projection = nn.Linear(1536, self.biogpt_config.hidden_size, False)

        self.embed_positions = biogpt.embed_positions
        self.embed_tokens = biogpt.embed_tokens

        self.layers = nn.ModuleList([])
        for i in range(len(biogpt.layers)):
            self.layers.append(
                nn.ModuleList(
                    [
                        GatedCrossAttentionBlock(
                            dim=self.biogpt_config.hidden_size,
                            dim_head=(
                                self.biogpt_config.hidden_size //
                                self.biogpt_config.num_attention_heads
                            ),
                            heads=self.biogpt_config.num_attention_heads,
                            ff_mult=4,
                            only_attend_immediate_media=True
                        ),
                        biogpt.layers[i],
                    ]
                )
            )

        self.layer_norm = biogpt.layer_norm
        self.gradient_checkpointing = checkpoint

    def decoder_forward(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_head_mask: torch.Tensor,
        past_key_value: torch.Tensor,
        output_attentions: torch.Tensor,
        use_cache: bool,
    ):
        return module(
            hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )

    def xattn_forward(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        image_latents: torch.Tensor,
    ):
        return module(hidden_states, image_latents)

    def forward(
        self,
        input_ids: torch.LongTensor,
        image_emb: torch.FloatTensor = None,
        image_pos: torch.LongTensor = None,
        attention_mask: torch.Tensor = None,
    ):
        output_attentions = self.biogpt_config.output_attentions
        use_cache = self.biogpt_config.use_cache

        # retrieve input tokens
        input = input_ids
        input_shape = input.size()
        past_key_values_length = 0

        # embed input tokens
        inputs_embeds = self.embed_tokens(input) * self.embed_scale

        # prepare attention masks
        if attention_mask is None:
            attention_mask = torch.ones(
                (
                    inputs_embeds.shape[0],
                    inputs_embeds.shape[1] + past_key_values_length,
                ),
                dtype=torch.bool,
                device=inputs_embeds.device,
            )
        else:
            # Use the provided attention mask
            attention_mask = attention_mask.bool()

        # embed token positions
        positions = self.embed_positions(attention_mask, past_key_values_length)

        # create attention masks
        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )

        # make hidden states
        hidden_states = inputs_embeds + positions
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=False
        )

        # prepare image features
        if image_emb is not None:
            image_latents = self.aggregator(image_emb, image_pos)
            # Ensure image_latents has the right shape for projection
            # Should be [batch_size, seq_len, 1536] -> [batch_size, seq_len, hidden_size]
            image_latents = self.projection(image_latents)

        # loop over transformer layers
        for (xattn_layer, decoder_layer) in self.layers:
            if self.gradient_checkpointing:
                if image_emb is not None:
                    hidden_states = checkpoint(
                        self.xattn_forward,
                        xattn_layer,
                        hidden_states,
                        image_latents,
                        use_reentrant=False
                    )
                layer_outputs = checkpoint(
                    self.decoder_forward,
                    decoder_layer,
                    hidden_states,
                    attention_mask,
                    None,
                    None,
                    output_attentions,
                    use_cache,
                    use_reentrant=False
                )
            else:
                if image_emb is not None:
                    hidden_states = xattn_layer(hidden_states, image_latents)
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    layer_head_mask=None,
                    past_key_value=None,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]
        hidden_states = self.layer_norm(hidden_states)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
            cross_attentions=None,
        )


class HistoGPTForCausalLM(nn.Module):
    """
    The HistoGPT wrapper for causal language modeling
    """
    def __init__(self, aggregator: nn.Module, biogpt: nn.Module, checkpoint: bool):
        super().__init__()
        requires_grad(biogpt, False)
        requires_grad(aggregator, False)
        self.config = biogpt.config  # Store config for LoRA compatibility
        self.histogpt = HistoGPTModel(aggregator, biogpt.biogpt, checkpoint)
        self.output_projection = biogpt.output_projection

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor = None,
        image_emb: torch.FloatTensor = None,
        image_pos: torch.LongTensor = None,
        **kwargs
    ):
        # compute output with llm
        outputs = self.histogpt(input_ids, image_emb, image_pos, attention_mask)

        # extract hidden states
        sequence_output = outputs[0]

        # return logits from head
        prediction_scores = self.output_projection(sequence_output)

        return CausalLMOutputWithCrossAttentions(
            loss=None,
            logits=prediction_scores,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        """
        Prepare inputs for generation. Required for LoRA compatibility.
        """
        model_inputs = {"input_ids": input_ids}
        
        # Add image features if provided
        if "image_emb" in kwargs:
            model_inputs["image_emb"] = kwargs["image_emb"]
        if "image_pos" in kwargs:
            model_inputs["image_pos"] = kwargs["image_pos"]
            
        return model_inputs
