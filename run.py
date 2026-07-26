#!/usr/bin/env python3
"""Entry point. Everything real lives in the tekdromo package."""
import os
import sys

os.environ.setdefault("OPENBLAS_CORETYPE", "ARMV8")   # numpy SIGILLs without it
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tekdromo.app import main

if __name__ == "__main__":
    main()
