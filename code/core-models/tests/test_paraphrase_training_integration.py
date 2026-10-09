import importlib.util
import re
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import torch
    import transformers  # noqa: F401
except ImportError:
    torch = None


if torch is not None:
    from core_bert.benchmark_data import BenchmarkExample
    from core_bert.structured_paraphrases import build_structured_partition

    spec = importlib.util.spec_from_file_location(
        "train_addressed_paraphrase_test", ROOT / "src" / "train_addressed.py"
    )
    trainer = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(trainer)


class FakeTokenizer:
    cls_token_id = 101
    pad_token_id = 0
    unk_token_id = 100

    def __init__(self):
        self.command_texts = []

    def convert_tokens_to_ids(self, token):
        return 777 if token == "[DO]" else self.unk_token_id

    def __call__(
        self, text, *, add_special_tokens=True, return_offsets_mapping=False,
        truncation=False, padding=False, max_length=None, return_tensors=None,
    ):
        if isinstance(text, str):
            matches = list(re.finditer(r"\S+", text))
            if not return_offsets_mapping:
                self.command_texts.append(text)
            result = {"input_ids": [1000 + index for index in range(len(matches))]}
            if return_offsets_mapping:
                result["offset_mapping"] = [
                    (match.start(), match.end()) for match in matches
                ]
            return result
        rows = []
        for value in text:
            tokens = [20 + index for index, _ in enumerate(value.split())]
            if add_special_tokens:
                tokens = [101, *tokens, 102]
            if max_length is not None:
                tokens = tokens[:max_length]
            rows.append(tokens)
        width = max_length if padding == "max_length" else max(map(len, rows))
        ids = [row + [0] * (width - len(row)) for row in rows]
        mask = [[int(token != 0) for token in row] for row in ids]
        return {
            "input_ids": torch.tensor(ids),
            "attention_mask": torch.tensor(mask),
        }


def fixture():
    passage = "Variable X is high."
    return {
        "id": "fixture-1",
        "source": "fixture",
        "source_id": "fixture-1",
        "structure_kind": "dag",
        "graph": {"nodes": ["X"], "edges": []},
        "chain": None,
        "factual": {"passage": passage, "question": "What is X?"},
        "intervention": {
            "kind": "value_set",
            "target": "X",
            "target_text": "X",
            "target_span": [9, 10],
            "value": "low",
            "value_token": "low",
            "formal": "do(X=low)",
            "text": "Original instruction.",
            "replacement_span": None,
        },
        "intervened": {"answer": "less"},
        "provenance": {"fixture": True},
    }


@unittest.skipUnless(torch is not None, "torch/transformers unavailable")
class ProductionParaphraseCollatorTests(unittest.TestCase):
    def test_train_and_validation_collators_use_only_their_banks(self):
        record = fixture()
        partition, _ = build_structured_partition(
            {"train": [record], "validation": []}, seed=13
        )
        operator = "value_set"
        intervention = partition.assignments[0].intervention_key
        example = BenchmarkExample(
            record, record["id"], record["source"], "less", "value_set", "g", "w"
        )
        train_tokenizer = FakeTokenizer()
        validation_tokenizer = FakeTokenizer()
        train_collator = trainer.ProductionCollator(
            train_tokenizer, 64, 30, variable_outputs=False,
            paraphrase_partition=partition, split="train",
        )
        validation_collator = trainer.ProductionCollator(
            validation_tokenizer, 64, 30, variable_outputs=False,
            paraphrase_partition=partition, split="validation",
        )
        train_batch = train_collator([example])
        validation_batch = validation_collator([example])
        train_texts = {item.text for item in partition.texts_for(operator, intervention, "train")}
        validation_texts = {
            item.text for item in partition.texts_for(operator, intervention, "validation")
        }
        self.assertIn(train_tokenizer.command_texts[-1], train_texts)
        self.assertIn(validation_tokenizer.command_texts[-1], validation_texts)
        self.assertTrue(train_texts.isdisjoint(validation_texts))
        self.assertEqual(train_batch["record_ids"], [record["id"]])
        self.assertEqual(validation_batch["record_ids"], [record["id"]])
        self.assertEqual(record["intervention"]["text"], "Original instruction.")

    def test_test_collator_is_rejected_before_batching(self):
        partition, _ = build_structured_partition(
            {"train": [fixture()], "validation": []}
        )
        with self.assertRaisesRegex(PermissionError, "held-out test"):
            trainer.ProductionCollator(
                FakeTokenizer(), 64, 30, variable_outputs=False,
                paraphrase_partition=partition, split="test",
            )


if __name__ == "__main__":
    unittest.main()
