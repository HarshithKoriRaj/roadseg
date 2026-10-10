def test_environment_has_required_packages():
    import torch, torchvision, numpy, scipy, skimage, albumentations, networkx
    assert torch.__version__.startswith("2.")
    assert hasattr(skimage.morphology, "skeletonize")
