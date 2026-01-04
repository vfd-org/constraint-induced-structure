"""
CISE CLI entrypoint.

Usage:
    python -m cise run --config configs/default.yaml --out outputs/run_001
    python -m cise sweep --beta 0 0.5 1 2 --out outputs/run_002
    python -m cise report --in outputs/run_001
"""

from cise.cli import main

if __name__ == "__main__":
    main()
