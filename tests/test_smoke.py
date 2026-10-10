def test_environment_has_required_packages():
    import torch, torchvision, numpy, scipy, skimage, albumentations, networkx
    assert torch.__version__.startswith("2.")
    assert hasattr(skimage.morphology, "skeletonize")


def test_core_packages_resolve_from_conda_env_not_deps():
    import numpy, scipy, PIL, networkx, yaml
    for module in (numpy, scipy, PIL, networkx, yaml):
        assert "/.deps/" not in module.__file__, (
            f"{module.__name__} loaded from {module.__file__}, which is inside .deps/; "
            f"prune {module.__name__} from .deps so the conda env's copy is used instead"
        )
