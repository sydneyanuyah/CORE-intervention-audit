"""Split-safe benchmark loading and group-preserving distributed sampling."""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ALLOWED_SPLITS = {"train", "validation", "test"}


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest(prefix: str, value: Any) -> str:
    payload = f"{prefix}:{_stable_json(value)}".encode("utf-8")
    return f"{prefix}:{hashlib.sha256(payload).hexdigest()[:20]}"


def _source_groups(record: Mapping[str, Any]) -> tuple[str, str]:
    """Return stable graph/world groups, preserving explicit IDs when present."""

    source = str(record["source"])
    if record.get("graph_id") is not None:
        graph_group = f"{source}:graph:{record['graph_id']}"
    elif source == "ccrgb":
        context = str(record["source_id"]).split(":", 1)[0]
        graph_group = f"ccrgb:context:{context}"
    elif source == "wiqa":
        question_graph = str(record["source_id"]).split("#", 1)[0]
        graph_group = f"wiqa:{question_graph}"
    elif source == "com2" and record.get("chain"):
        graph_group = _digest("com2-root", record["chain"][0])
    else:
        structure = record.get("graph") if record.get("graph") is not None else record.get("chain")
        graph_group = _digest(f"{source}-graph", structure)

    if record.get("world_id") is not None:
        world_group = f"{source}:world:{record['world_id']}"
    elif source == "ccrgb":
        world_group = graph_group
    elif source == "wiqa":
        world_group = graph_group
    elif source == "com2" and record.get("chain"):
        world_group = _digest("com2-chain", record["chain"])
    else:
        factual = record.get("factual") or {}
        world_group = _digest(
            f"{source}-world",
            {
                "graph_group": graph_group,
                "passage": factual.get("passage"),
                "state": factual.get("state"),
            },
        )
    return graph_group, world_group


@dataclass(frozen=True)
class BenchmarkExample:
    """One raw record plus the metadata needed for analysis and sharding."""

    record: dict[str, Any]
    record_id: str
    source: str
    label: Any
    intervention_kind: str
    graph_group_id: str
    world_group_id: str


class BenchmarkDataset(Sequence[BenchmarkExample]):
    def __init__(self, examples: Sequence[BenchmarkExample]):
        self.examples = tuple(examples)

    def __getitem__(self, index: int) -> BenchmarkExample:
        return self.examples[index]

    def __len__(self) -> int:
        return len(self.examples)

    @property
    def records(self) -> list[dict[str, Any]]:
        """Return raw records for an addressed-data collator."""

        return [example.record for example in self.examples]


def _manifest_ids(splits_dir: Path, split: str) -> set[str]:
    paths = sorted(splits_dir.glob(f"*.{split}.txt"))
    if not paths:
        raise ValueError(f"no split manifests found for {split!r} in {splits_dir}")
    selected: set[str] = set()
    for path in paths:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            record_id = line.strip()
            if not record_id or record_id.startswith("#"):
                continue
            if record_id in selected:
                raise ValueError(f"duplicate split ID {record_id!r} at {path}:{line_number}")
            selected.add(record_id)
    return selected


def load_benchmark_split(
    data_root: Path,
    split: str,
    *,
    sources: Sequence[str] | None = None,
    final_eval: bool = False,
) -> BenchmarkDataset:
    """Load accepted records assigned to one split.

    ``test`` is inaccessible unless the caller explicitly passes
    ``final_eval=True``. Rejected JSONL files are never considered. Selecting
    ``com2`` includes both accepted Com2 record files.
    """

    if split not in ALLOWED_SPLITS:
        raise ValueError(f"unsupported split {split!r}")
    if split == "test" and not final_eval:
        raise PermissionError("test split requires explicit final_eval=True")
    selected_sources = set(sources) if sources is not None else None
    if selected_sources is not None and (not selected_sources or not all(selected_sources)):
        raise ValueError("sources must contain non-empty source names")

    selected_ids = _manifest_ids(data_root / "splits", split)
    found_sources: set[str] = set()
    seen_ids: set[str] = set()
    examples: list[BenchmarkExample] = []
    for path in sorted((data_root / "records").glob("*.jsonl")):
        if path.name.endswith(".rejected.jsonl"):
            continue
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number}: record must be an object")
                source = value.get("source")
                if isinstance(source, str):
                    found_sources.add(source)
                if selected_sources is not None and source not in selected_sources:
                    continue
                if not isinstance(source, str) or not source:
                    raise ValueError(f"{path}:{line_number}: source must be a non-empty string")
                record_id = value.get("id")
                if record_id not in selected_ids:
                    continue
                if not isinstance(record_id, str) or not record_id:
                    raise ValueError(f"{path}:{line_number}: record ID must be a non-empty string")
                if record_id in seen_ids:
                    raise ValueError(f"duplicate accepted record ID {record_id!r}")
                intervention = value.get("intervention")
                intervened = value.get("intervened")
                if not isinstance(intervention, dict) or not isinstance(intervened, dict):
                    raise ValueError(f"{path}:{line_number}: malformed intervention/intervened object")
                graph_group, world_group = _source_groups(value)
                examples.append(
                    BenchmarkExample(
                        record=value,
                        record_id=record_id,
                        source=source,
                        label=intervened.get("answer"),
                        intervention_kind=str(intervention.get("kind")),
                        graph_group_id=graph_group,
                        world_group_id=world_group,
                    )
                )
                seen_ids.add(record_id)

    if selected_sources is not None:
        unknown = selected_sources - found_sources
        if unknown:
            raise ValueError(f"requested sources not found: {sorted(unknown)}")
    if not examples:
        raise ValueError(f"no accepted records found for split={split!r}, sources={sources!r}")
    examples.sort(key=lambda example: example.record_id)
    return BenchmarkDataset(examples)


def deterministic_smoke_subset(
    dataset: Sequence[BenchmarkExample],
    *,
    per_stratum: int = 1,
    seed: int = 0,
) -> BenchmarkDataset:
    """Select stable examples from every source/label/intervention-kind stratum."""

    if per_stratum < 1:
        raise ValueError("per_stratum must be positive")
    strata: dict[tuple[str, str, str], list[BenchmarkExample]] = defaultdict(list)
    for example in dataset:
        label_key = _stable_json(example.label)
        if example.source == "com2" and isinstance(example.label, str):
            choice = re.match(r"^\s*([A-D])(?:\)|\.)", example.label, re.IGNORECASE)
            label_key = f"choice:{choice.group(1).upper()}" if choice else "open_text"
        strata[(example.source, label_key, example.intervention_kind)].append(example)
    selected: list[BenchmarkExample] = []
    for key in sorted(strata):
        ranked = sorted(
            strata[key],
            key=lambda example: hashlib.sha256(
                f"{seed}:{example.record_id}".encode("utf-8")
            ).digest(),
        )
        selected.extend(ranked[:per_stratum])
    return BenchmarkDataset(sorted(selected, key=lambda example: example.record_id))


class GroupDistributedSampler:
    """Deterministically shard intact graph/world groups across DDP ranks.

    Groups are greedily balanced by example count. With ``pad_to_equal=True``,
    only examples already owned by a rank are repeated so every rank receives
    the same number of samples; a group is never split between ranks.
    Evaluation should use ``pad_to_equal=False`` to avoid duplicate metrics.
    """

    def __init__(
        self,
        dataset: Sequence[BenchmarkExample],
        *,
        rank: int,
        world_size: int,
        group_by: str = "graph",
        seed: int = 0,
        shuffle: bool = True,
        pad_to_equal: bool = True,
    ) -> None:
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError("rank must satisfy 0 <= rank < world_size")
        if group_by not in {"graph", "world"}:
            raise ValueError("group_by must be 'graph' or 'world'")
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size
        self.group_by = group_by
        self.seed = seed
        self.shuffle = shuffle
        self.pad_to_equal = pad_to_equal
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _all_shards(self) -> list[list[int]]:
        attribute = f"{self.group_by}_group_id"
        groups: dict[str, list[int]] = defaultdict(list)
        for index, example in enumerate(self.dataset):
            groups[getattr(example, attribute)].append(index)
        if self.pad_to_equal and len(groups) < self.world_size:
            raise ValueError("fewer groups than DDP ranks; group-preserving sharding is impossible")
        items = list(groups.items())
        rng = random.Random(self.seed + self.epoch)
        if self.shuffle:
            rng.shuffle(items)
            for _, indices in items:
                rng.shuffle(indices)
        items.sort(key=lambda item: len(item[1]), reverse=True)
        shards = [[] for _ in range(self.world_size)]
        loads = [0] * self.world_size
        for _, indices in items:
            destination = min(range(self.world_size), key=lambda rank: (loads[rank], rank))
            shards[destination].extend(indices)
            loads[destination] += len(indices)
        if self.pad_to_equal:
            target = max(map(len, shards), default=0)
            for shard in shards:
                original = list(shard)
                if not original:
                    raise ValueError("cannot pad an empty rank shard")
                shard.extend(original[index % len(original)] for index in range(target - len(shard)))
        return shards

    def __iter__(self) -> Iterator[int]:
        return iter(self._all_shards()[self.rank])

    def __len__(self) -> int:
        return len(self._all_shards()[self.rank])
