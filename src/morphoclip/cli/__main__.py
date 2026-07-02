"""Enable ``python -m morphoclip.cli`` (used by torchrun for distributed training)."""

from morphoclip.cli import app

if __name__ == "__main__":
    app()
