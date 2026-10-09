"""Split BERT reader integrating T2 inline and T3 span addressing.

The reader owns representation plumbing only.  Editors are injected modules,
so the canonical O2/O3 implementations remain shared with the server code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from .t2_inline import build_t2b_masks, pool_t2_instruction
from .t3_pointer import PointerOutput, build_span_instruction, mean_pool_span


T2Mode = Literal["t2a", "t2b"]
T3Mode = Literal["t3a", "t3b"]


@dataclass(frozen=True)
class AddressedReaderOutput:
    logits: torch.Tensor
    edited_slots: torch.Tensor
    instruction: torch.Tensor
    pointer_probabilities: torch.Tensor | None = None


class AddressedWorldReader(nn.Module):
    """BERT split at an edit layer with learned world and output-query slots."""

    def __init__(
        self,
        bert: nn.Module,
        node_count: int,
        *,
        split_layer: int,
        world_slot_count: int = 16,
        label_count: int = 2,
    ):
        super().__init__()
        layer_count = len(bert.encoder.layer)
        if not 0 < split_layer < layer_count:
            raise ValueError("split_layer must leave encoder layers before and after the edit")
        if node_count < 1 or world_slot_count < 1:
            raise ValueError("node and world-slot counts must be positive")
        self.bert = bert
        # Transformers may default a directly constructed BERT to SDPA.  The
        # T2 directional [batch, query, key] masks require the eager path.
        if hasattr(bert.config, "_attn_implementation"):
            bert.config._attn_implementation = "eager"
        self.hidden_size = int(bert.config.hidden_size)
        self.split_layer = split_layer
        self.world_slot_count = world_slot_count
        self.world_slots = nn.Parameter(torch.randn(world_slot_count, self.hidden_size) * 0.02)
        self.query_slots = nn.Parameter(torch.randn(node_count, self.hidden_size) * 0.02)
        self.answer_head = nn.Linear(self.hidden_size, label_count)

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        node_count: int,
        *,
        split_layer: int,
        world_slot_count: int = 16,
        label_count: int = 2,
        **model_kwargs,
    ) -> "AddressedWorldReader":
        """Load canonical Hugging Face BERT with eager 3D-mask attention."""

        from transformers import AutoModel

        model_kwargs.setdefault("attn_implementation", "eager")
        bert = AutoModel.from_pretrained(model_name, **model_kwargs)
        if getattr(bert.config, "model_type", None) != "bert":
            raise ValueError("AddressedWorldReader currently requires a BERT encoder")
        return cls(
            bert,
            node_count,
            split_layer=split_layer,
            world_slot_count=world_slot_count,
            label_count=label_count,
        )

    @staticmethod
    def _additive(allowed: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        bias = torch.zeros(allowed.shape, dtype=dtype, device=allowed.device)
        return bias.masked_fill(~allowed, torch.finfo(dtype).min)[:, None, :, :]

    @staticmethod
    def _run_layers(layers, hidden: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
        bias = AddressedWorldReader._additive(allowed, hidden.dtype)
        for layer in layers:
            output = layer(hidden, attention_mask=bias)
            # Transformers 4.x returns a tuple from BertLayer; 5.x returns the
            # hidden tensor directly.  Support both pinned and current servers.
            hidden = output[0] if isinstance(output, (tuple, list)) else output
        return hidden

    @staticmethod
    def _full_attention(valid: torch.Tensor) -> torch.Tensor:
        valid = valid.bool()
        return valid[:, :, None] & valid[:, None, :]

    def _embed_with_slots(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if input_ids.shape != attention_mask.shape or input_ids.ndim != 2:
            raise ValueError("input_ids and attention_mask must share shape [batch, sequence]")
        hidden = self.bert.embeddings(input_ids=input_ids)
        slots = self.world_slots[None, :, :].expand(input_ids.shape[0], -1, -1)
        hidden = torch.cat([hidden, slots], dim=1)
        slot_valid = torch.ones(
            input_ids.shape[0],
            self.world_slot_count,
            dtype=torch.bool,
            device=input_ids.device,
        )
        return hidden, torch.cat([attention_mask.bool(), slot_valid], dim=1)

    def _decode(
        self,
        trunk_hidden: torch.Tensor,
        trunk_valid: torch.Tensor,
        edited_slots: torch.Tensor,
        hidden_instruction_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = trunk_hidden.clone()
        hidden[:, -self.world_slot_count :] = edited_slots
        queries = self.query_slots[None, :, :].expand(hidden.shape[0], -1, -1)
        hidden = torch.cat([hidden, queries], dim=1)
        query_valid = torch.ones(
            hidden.shape[0], len(self.query_slots), dtype=torch.bool, device=hidden.device
        )
        valid = torch.cat([trunk_valid, query_valid], dim=1)
        allowed = self._full_attention(valid)
        if hidden_instruction_tokens is not None:
            if hidden_instruction_tokens.shape != trunk_valid.shape:
                raise ValueError("instruction mask must cover the pre-query trunk")
            # Instruction states remain in the tensor but are never visible as
            # keys in any post-edit layer. Query-side masking is unnecessary:
            # self-attention updates all positions simultaneously, and these
            # positions remain hidden as keys at every subsequent layer.
            hidden_keys = torch.cat(
                [
                    hidden_instruction_tokens.bool(),
                    torch.zeros_like(query_valid),
                ],
                dim=1,
            )
            allowed &= ~hidden_keys[:, None, :]
        hidden = self._run_layers(
            self.bert.encoder.layer[self.split_layer :], hidden, allowed
        )
        return self.answer_head(hidden[:, -len(self.query_slots) :])

    def forward_t2(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        command_tokens: torch.Tensor,
        do_tokens: torch.Tensor,
        editor: nn.Module,
        *,
        mode: T2Mode = "t2b",
        pooling: Literal["mean", "last"] = "mean",
        editor_enabled: bool = True,
    ) -> AddressedReaderOutput:
        """Run T2-a naive or T2-b encode-only through actual encoder layers."""

        if command_tokens.shape != input_ids.shape or do_tokens.shape != input_ids.shape:
            raise ValueError("T2 token masks must match input_ids")
        hidden, valid = self._embed_with_slots(input_ids, attention_mask)
        input_length = input_ids.shape[1]
        slot_tokens = torch.zeros_like(valid)
        slot_tokens[:, input_length:] = True
        command_extended = torch.cat(
            [command_tokens.bool(), torch.zeros_like(slot_tokens[:, input_length:])], dim=1
        )
        do_extended = torch.cat(
            [do_tokens.bool(), torch.zeros_like(slot_tokens[:, input_length:])], dim=1
        )

        if mode == "t2a":
            pre_allowed = self._full_attention(valid)
            hidden_instruction = None
        elif mode == "t2b":
            masks = build_t2b_masks(
                valid, command_extended, do_extended, slot_tokens
            )
            pre_allowed = masks.pre_edit
            hidden_instruction = masks.instruction_tokens
            # A directional isolation barrier is required across multiple
            # pre-edit layers: otherwise passage states can read the command
            # in layer 1 and relay it to slots in layer 2. Command states may
            # still attend to and ground themselves in the passage.
            non_instruction_queries = ~hidden_instruction
            pre_allowed &= ~(
                non_instruction_queries[:, :, None]
                & hidden_instruction[:, None, :]
            )
        else:
            raise ValueError(f"unknown T2 mode: {mode}")

        trunk = self._run_layers(
            self.bert.encoder.layer[: self.split_layer], hidden, pre_allowed
        )
        instruction = pool_t2_instruction(
            trunk[:, :input_length], command_tokens, do_tokens, pooling
        )
        slots = trunk[:, -self.world_slot_count :]
        edited = editor(slots, instruction) if editor_enabled else slots
        if edited.shape != slots.shape:
            raise ValueError("T2 editor must return the world-slot tensor shape")
        logits = self._decode(trunk, valid, edited, hidden_instruction)
        return AddressedReaderOutput(logits, edited, instruction)

    def forward_t3(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_span: torch.Tensor,
        editor: nn.Module,
        *,
        value_embedding: torch.Tensor | None = None,
        replacement_span: torch.Tensor | None = None,
        mode: T3Mode = "t3b",
        slot_mask: torch.Tensor | None = None,
    ) -> AddressedReaderOutput:
        """Run T3-a conditioning or T3-b textual-span pointer selection."""

        hidden, valid = self._embed_with_slots(input_ids, attention_mask)
        trunk = self._run_layers(
            self.bert.encoder.layer[: self.split_layer],
            hidden,
            self._full_attention(valid),
        )
        text_hidden = trunk[:, : input_ids.shape[1]]
        address = mean_pool_span(text_hidden, target_span)
        instruction = build_span_instruction(
            text_hidden,
            target_span,
            value_embedding=value_embedding,
            replacement_span=replacement_span,
        )
        slots = trunk[:, -self.world_slot_count :]
        probabilities = None
        if mode == "t3a":
            edited = editor(slots, instruction)
        elif mode == "t3b":
            result = editor(slots, address, instruction, slot_mask)
            if not isinstance(result, PointerOutput):
                # Structural typing permits a server-side shared pointer
                # implementation without requiring this exact class.
                try:
                    edited = result.edited_slots
                    probabilities = result.probabilities
                except AttributeError as error:
                    raise TypeError("T3-b editor must return edited_slots and probabilities") from error
            else:
                edited = result.edited_slots
                probabilities = result.probabilities
        else:
            raise ValueError(f"unknown T3 mode: {mode}")
        if edited.shape != slots.shape:
            raise ValueError("T3 editor must return the world-slot tensor shape")
        logits = self._decode(trunk, valid, edited)
        return AddressedReaderOutput(logits, edited, instruction, probabilities)
