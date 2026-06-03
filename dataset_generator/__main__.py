"""Allow running as: python -m dataset_generator"""

from dataset_generator.cli import main

if __name__ == "__main__":
    import sys
    sys.exit(main())
