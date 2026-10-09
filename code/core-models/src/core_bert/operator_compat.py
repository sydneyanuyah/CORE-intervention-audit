"""Run historical CORE split-reader scripts with compatible BERT attention."""

from __future__ import annotations

import runpy
import sys
import importlib
from pathlib import Path


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit("usage: operator_compat.py TARGET_SCRIPT [TARGET_ARGS...]")
    target = Path(sys.argv[1]).resolve()
    target_args = sys.argv[2:]
    # Executing this file directly places core_bert/ on sys.path, where our
    # statistics.py would shadow Python's standard-library statistics module.
    this_directory = str(Path(__file__).resolve().parent)
    sys.path = [entry for entry in sys.path if str(Path(entry or ".").resolve()) != this_directory]
    import transformers

    original = transformers.AutoModel.from_pretrained

    def eager_from_pretrained(*args, **kwargs):
        kwargs.setdefault("attn_implementation", "eager")
        return original(*args, **kwargs)

    transformers.AutoModel.from_pretrained = eager_from_pretrained
    sys.path.insert(0, str(target.parent))
    if target.name == "run_closing_controls.py":
        core = importlib.import_module("core_components")

        def run_layers(layers, hidden, bias):
            for layer in layers:
                output = layer(hidden, attention_mask=bias)
                hidden = output[0] if isinstance(output, (tuple, list)) else output
            return hidden

        def trunk(self, input_ids, attention_mask):
            hidden = self.bert.embeddings(input_ids=input_ids)
            world = self.world_token_embeddings[None].expand(len(hidden), -1, -1)
            hidden = __import__("torch").cat([hidden, world], dim=1)
            world_mask = __import__("torch").ones(
                len(hidden), self.max_world_tokens,
                dtype=attention_mask.dtype, device=attention_mask.device,
            )
            expanded = __import__("torch").cat([attention_mask, world_mask], dim=1)
            bias = self.attention_bias(expanded, hidden.dtype)
            return run_layers(self.bert.encoder.layer[: self.split_layer], hidden, bias), expanded

        def decode(self, world_hidden, world_mask):
            queries = self.query_embeddings[None].expand(len(world_hidden), -1, -1)
            hidden = __import__("torch").cat([world_hidden, queries], dim=1)
            query_mask = __import__("torch").ones(
                len(hidden), len(self.query_embeddings),
                dtype=world_mask.dtype, device=world_mask.device,
            )
            mask = __import__("torch").cat([world_mask, query_mask], dim=1)
            bias = self.attention_bias(mask, hidden.dtype)
            hidden = run_layers(self.bert.encoder.layer[self.split_layer :], hidden, bias)
            return self.answer_head(hidden[:, -len(self.query_embeddings) :])

        core.SharedWorldReader.trunk = trunk
        core.SharedWorldReader.decode = decode
    sys.argv = [str(target), *target_args]
    runpy.run_path(str(target), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
