from pathlib import Path


def test_dockerfile():
    """Test that we're copying everything we should be.

    This test isn't particularly robust, so expect it to break if we start
    copying things differently in the Dockerfile."""
    parent_dir = Path(__file__).parent
    dockerfile = (parent_dir / "Dockerfile").read_text()
    dirs = {file.name for file in parent_dir.iterdir() if file.is_dir()}
    dirs.difference_update({"__pycache__", ".pytest_cache", "venv", "kernel_spec"})
    for line in dockerfile.splitlines():
        if line.startswith("COPY"):
            parts = line.split()
            source, dest = (arg for arg in parts[1:] if not arg.startswith("--"))
            dirs.discard(source)

    assert not dirs, f"Dockerfile does not copy: {dirs}"
