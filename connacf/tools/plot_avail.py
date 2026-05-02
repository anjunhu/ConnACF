#!/usr/bin/env python3
"""
Plot availability / resource-exhaustion metrics for Corba attacks.

Plots a single custom metric per turn (dot-notation path into turn JSON).

Usage:
    python tools/plot_avail.py --last_epoch 150 \\
        --metric resources.corba_blocking_rate \\
        --title "Corba x ConnaCF x MovieLens100 (Blocking Rate)" \\
        --output "../figure/corba_blocking.png" \\
        --connacf_dirs \\
            attack_output/corba/corba_canonical_2cand/ml-100k-100user-dense/task_0 \\
        --labels "Medium (2c)"
"""

# plot_avail is identical in structure to plot_privacy — resource metrics are
# also single-panel custom metric plots. We reuse the same implementation.

from plot_privacy import plot_privacy, main as _privacy_main
from _plot_common import base_parser, load_datasets
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = base_parser('Plot availability/resource metrics (Corba)')
    parser.add_argument('--metric', type=str, required=True,
                        help='Dot-notation metric path, e.g. resources.corba_blocking_rate')
    args = parser.parse_args()
    datasets = load_datasets(args, custom_metric=args.metric)
    valid = [d for d in datasets if d['turns']]
    if not valid:
        logger.error("No valid datasets found.")
        return
    plot_privacy(valid, args.metric, args.output, args.title)


if __name__ == '__main__':
    main()
