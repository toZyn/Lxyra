import pytest


def test_compact_detector_has_same_output_contract_as_v2():
    torch = pytest.importorskip("torch")
    from lxyra.training.compact_model import LxyraCompact
    from lxyra.training.detector_model import LxyraV2

    torch.set_num_threads(2)
    original = LxyraV2().eval()
    compact = LxyraCompact().eval()
    assert sum(p.numel() for p in compact.parameters()) < 0.10 * sum(
        p.numel() for p in original.parameters()
    )
    with torch.inference_mode():
        for side in (320, 448, 640):
            shapes = [part.shape for part in compact(torch.rand(1, 3, side, side))]
            assert shapes == [
                (1, 10, side // 4, side // 4),
                (1, 2, side // 4, side // 4),
                (1, 2, side // 4, side // 4),
            ]


def test_compact_backpropagation_updates_weights():
    torch = pytest.importorskip("torch")
    from lxyra.training.compact_model import LxyraCompact
    from lxyra.training.detector_model import loss, target

    torch.set_num_threads(2)
    torch.manual_seed(211)
    model = LxyraCompact()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    annotations = [{"category": "button", "bbox": [40, 60, 95, 35]}]
    labels = target(annotations, (1.0, 1.0, 0.0, 0.0), size=448)
    labels = tuple(t.unsqueeze(0) for t in labels)
    before = model.cls.weight.detach().clone()
    value, _ = loss(model(torch.rand(1, 3, 448, 448)), labels)
    assert torch.isfinite(value).item()
    optimizer.zero_grad()
    value.backward()
    optimizer.step()
    assert not torch.equal(before, model.cls.weight.detach())
