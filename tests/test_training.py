import pytest


def test_randomly_initialized_detector_backward():
    torch = pytest.importorskip("torch")
    from lxyra.training.detector_model import LxyraV2

    torch.set_num_threads(2)
    net = LxyraV2()
    images = torch.rand(1, 3, 448, 448)
    output = net(images)
    assert len(output) == 3
    assert output[0].shape[0] == 1
    output[0].mean().backward()
    assert any(p.grad is not None for p in net.parameters())


def test_text_mask_model_shapes():
    torch = pytest.importorskip("torch")
    from lxyra.training.text_model import TextDetector

    torch.set_num_threads(2)
    net = TextDetector()
    output = net(torch.rand(1, 3, 448, 448))
    assert tuple(output.shape) == (1, 1, 112, 112)


def test_roi_refiner_shapes():
    torch = pytest.importorskip("torch")
    from lxyra.training.refiner_model import TextRefiner

    torch.set_num_threads(2)
    net = TextRefiner()
    output = net(torch.rand(2, 3, 96, 96))
    assert tuple(output.shape) == (2, 5)
