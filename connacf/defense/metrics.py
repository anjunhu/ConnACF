"""Defense metrics for tracking and reporting detection performance.

Computes precision, recall, and F1-score by comparing detected attackers
against ground-truth attacker indices. Tracks results over time and
supports JSON persistence and matplotlib visualization.
"""

import json
import logging
from typing import Dict, List, Optional

from connacf.defense.detector import DetectionResult

logger = logging.getLogger(__name__)


class DefenseMetrics:
    """Tracks and reports defense detection performance.

    Records detection results per round, computes precision/recall/F1
    for each detection, and provides aggregated summaries. Supports
    saving metrics to JSON and generating detection accuracy plots.

    Args:
        ground_truth_attackers: Set of agent indices that are true attackers.
    """

    def __init__(self, ground_truth_attackers: set):
        self.ground_truth_attackers: set = set(ground_truth_attackers)
        self._history: List[Dict] = []

    def record_detection(self, detection_result: DetectionResult, round_idx: int):
        """Record detection result and compute precision/recall/F1.

        Computes metrics by comparing the detected attacker set against
        ground truth, then stores them on the DetectionResult and in
        the internal history.

        Args:
            detection_result: Result from DefenseDetector.detect().
            round_idx: Training round index for this detection.
        """
        detected = set(detection_result.detected_attacker_indices)
        ground_truth = self.ground_truth_attackers

        true_positives = len(detected & ground_truth)

        # Precision: |D∩G| / |D|
        if len(detected) > 0:
            precision = true_positives / len(detected)
        else:
            precision = 0.0

        # Recall: |D∩G| / |G|
        if len(ground_truth) > 0:
            recall = true_positives / len(ground_truth)
        else:
            recall = 0.0

        # F1: 2 * precision * recall / (precision + recall)
        if (precision + recall) > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        # Store metrics on the detection result
        detection_result.precision = precision
        detection_result.recall = recall
        detection_result.f1 = f1

        # Record in history
        record = {
            "round_idx": round_idx,
            "detected_attacker_indices": detection_result.detected_attacker_indices,
            "all_scores": detection_result.all_scores,
            "defense_mode": detection_result.defense_mode,
            "timestamp": detection_result.timestamp,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        self._history.append(record)

        logger.info(
            "Round %d metrics: precision=%.3f, recall=%.3f, F1=%.3f",
            round_idx,
            precision,
            recall,
            f1,
        )

    def get_summary(self) -> dict:
        """Return aggregated metrics across all recorded rounds.

        Returns:
            Dictionary with ground_truth_attackers, num_detections,
            average precision/recall/F1, and per-round history.
        """
        if not self._history:
            return {
                "ground_truth_attackers": sorted(self.ground_truth_attackers),
                "num_detections": 0,
                "avg_precision": 0.0,
                "avg_recall": 0.0,
                "avg_f1": 0.0,
                "history": [],
            }

        n = len(self._history)
        avg_precision = sum(r["precision"] for r in self._history) / n
        avg_recall = sum(r["recall"] for r in self._history) / n
        avg_f1 = sum(r["f1"] for r in self._history) / n

        return {
            "ground_truth_attackers": sorted(self.ground_truth_attackers),
            "num_detections": n,
            "avg_precision": avg_precision,
            "avg_recall": avg_recall,
            "avg_f1": avg_f1,
            "history": self._history,
        }

    def save_metrics(self, output_path: str):
        """Save metrics to JSON file.

        Writes the full summary (including per-round history) to the
        specified path. Scores are serialized as-is (float values).

        Args:
            output_path: File path for the JSON output.
        """
        summary = self.get_summary()
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("Saved defense metrics to %s", output_path)

    def plot_detection_over_time(self, output_path: str):
        """Generate detection accuracy plot over training rounds.

        Creates a matplotlib figure with precision, recall, and F1
        plotted against round index, and saves it to the given path.

        Args:
            output_path: File path for the plot image (e.g. .png).
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning(
                "matplotlib not available. Skipping detection plot generation."
            )
            return

        if not self._history:
            logger.warning("No detection history to plot.")
            return

        rounds = [r["round_idx"] for r in self._history]
        precisions = [r["precision"] for r in self._history]
        recalls = [r["recall"] for r in self._history]
        f1s = [r["f1"] for r in self._history]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(rounds, precisions, marker="o", label="Precision")
        ax.plot(rounds, recalls, marker="s", label="Recall")
        ax.plot(rounds, f1s, marker="^", label="F1")
        ax.set_xlabel("Training Round")
        ax.set_ylabel("Score")
        ax.set_title("Detection Accuracy Over Training Rounds")
        ax.set_ylim(-0.05, 1.05)
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        logger.info("Saved detection plot to %s", output_path)
