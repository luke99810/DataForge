"""Basic package import and initialization test."""


def test_import():
    """Ensure the package can be imported."""
    from dataset_generator import __version__, __author__

    assert __version__ == "1.0.0"
    assert isinstance(__author__, str)
