"""CLI entrypoint for the evaluation quality gate."""

import sys

from docextract.gates.quality_gate import main

if __name__ == "__main__":
    sys.exit(main())
