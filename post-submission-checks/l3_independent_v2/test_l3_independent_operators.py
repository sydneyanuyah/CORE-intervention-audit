import torch

from core_7b.l3_independent_operators import build_independent_operator, trainable_parameter_count


def test_operator_classes_and_parameters_are_distinct():
    modules = {method: build_independent_operator(method, 60, 32) for method in ("lora_matched", "loreft", "o2")}
    assert len({module.implementation_id for module in modules.values()}) == 3
    assert len({tuple(module.state_dict()) for module in modules.values()}) == 3
    assert all(trainable_parameter_count(module) > 0 for module in modules.values())


def test_methods_produce_distinct_nonzero_updates():
    torch.manual_seed(7)
    hidden = torch.randn(4, 32)
    command = torch.tensor([0, 1, 2, 3])
    outputs = []
    for method in ("lora_matched", "loreft", "o2"):
        module = build_independent_operator(method, 60, 32)
        with torch.no_grad():
            module.up.weight.normal_(0, 0.1)
        outputs.append(module(hidden, command))
    assert not torch.equal(outputs[0], outputs[1])
    assert not torch.equal(outputs[0], outputs[2])
    assert not torch.equal(outputs[1], outputs[2])
