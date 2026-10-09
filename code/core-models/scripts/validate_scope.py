#!/usr/bin/env python3
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from core_bert.registry import load_and_validate

scope = load_and_validate(Path(__file__).parents[1] / "registry" / "bert_scope.json")
print(f"valid: {len(scope['experiments'])} experiments, {len(scope['models'])} models")
