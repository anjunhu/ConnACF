#!/usr/bin/env python3
"""Entry point for the train-then-defend pipeline, callable from connacf/.

Delegates entirely to connacf.defense.train_then_defend.main().
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from connacf.defense.train_then_defend import main

if __name__ == "__main__":
    main()
