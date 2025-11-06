"""Command line utility that trains the RandomForest letter classifier.

The data set is expected to live in ``machine_learning/data.csv`` and be laid
out exactly like the capture script writes it: the first column contains the
letter label and the remaining 63 columns store the relative x, y and z
coordinates of the 21 Mediapipe hand landmarks.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import cross_val_score, train_test_split

# Default locations inside the repository. Users can point to custom files
# through the command line arguments.
DEFAULT_DATA_PATH = Path(__file__).resolve().parent / "data.csv"
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "model.pkl"
DEFAULT_REPORT_PATH = Path(__file__).resolve().parent / "model_report.json"

LOGGER = logging.getLogger(__name__)


@dataclass
class TrainingArtifacts:
    """Bundle the outputs of a training run for easy serialisation."""

    accuracy: float
    confusion_matrix: Iterable[Iterable[int]]
    labels: Iterable[str]
    classification_report: str
    cross_val: Optional[dict]


def configure_logging(verbose: bool) -> None:
    """Configure the root logger so feedback is readable from the CLI."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(format="%(levelname)s: %(message)s", level=level)


def load_dataset(csv_path: Path) -> Tuple[pd.Series, pd.DataFrame]:
    """Load the feature matrix and labels from *csv_path*."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    LOGGER.debug("Loading dataset from %s", csv_path)
    # The capture script writes a headerless CSV, therefore we pass header=None.
    data = pd.read_csv(csv_path, header=None)
    if data.shape[1] != 64:
        raise ValueError(
            f"Unexpected dataset shape {data.shape}; expected 64 columns (1 label + 63 features)."
        )

    labels = data.iloc[:, 0]
    features = data.iloc[:, 1:]
    return labels, features


def build_model(
    *,
    n_estimators: int,
    max_depth: Optional[int],
    random_state: int,
    n_jobs: int,
) -> RandomForestClassifier:
    """Create the RandomForest classifier with the desired hyper parameters."""
    LOGGER.debug(
        "Initialising RandomForest (n_estimators=%s, max_depth=%s, random_state=%s)",
        n_estimators,
        max_depth,
        random_state,
    )
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=n_jobs,
    )


def evaluate(
    model: RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> TrainingArtifacts:
    """Compute the main metrics for the trained *model*."""
    LOGGER.debug("Running evaluation on the held-out test set")
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    # Confusion matrix is cast to Python lists to make json serialisation trivial.
    matrix = confusion_matrix(y_test, predictions)
    labels = list(model.classes_)
    report = classification_report(y_test, predictions)

    LOGGER.info("Accuracy: %.2f%%", accuracy * 100)
    LOGGER.info("Classification report:\n%s", report)
    LOGGER.info("Confusion matrix:\n%s", pd.DataFrame(matrix, index=labels, columns=labels))

    return TrainingArtifacts(
        accuracy=accuracy,
        confusion_matrix=matrix.tolist(),
        labels=labels,
        classification_report=report,
        cross_val=None,
    )


def run_cross_validation(
    features: pd.DataFrame,
    labels: pd.Series,
    *,
    folds: int,
    model: RandomForestClassifier,
) -> dict:
    """Run a quick k-fold cross validation to estimate generalisation."""
    LOGGER.debug("Performing %s-fold cross validation", folds)
    scores = cross_val_score(model, features, labels, cv=folds)
    results = {
        "folds": folds,
        "scores": scores.tolist(),
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
    }
    LOGGER.info(
        "Cross-validation accuracy: %.2f%% +/- %.2f%%",
        results["mean"] * 100,
        results["std"] * 100,
    )
    return results


def save_model(model: RandomForestClassifier, path: Path) -> None:
    """Serialise *model* with joblib."""
    LOGGER.debug("Saving trained model to %s", path)
    joblib.dump(model, path)


def save_report(report_path: Path, artifacts: TrainingArtifacts) -> None:
    """Persist the collected metrics to *report_path* in JSON format."""
    LOGGER.debug("Writing training report to %s", report_path)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(asdict(artifacts), handle, indent=2)


def parse_args() -> argparse.Namespace:
    """Prepare and parse the CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Train the RandomForest classifier used by the GUI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH, help="CSV dataset to load")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH, help="Where to store the trained model")
    parser.add_argument(
        "--report-path",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Optional JSON file that records the metrics of the run",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Share of the dataset reserved for evaluation")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--n-estimators", type=int, default=300, help="Number of trees to grow in the forest")
    parser.add_argument("--max-depth", type=int, default=None, help="Maximum depth of the trees (None lets them grow)")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel jobs used by scikit-learn (-1 uses all cores)")
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=0,
        help="If >0, run k-fold cross validation with the provided number of folds",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging for debugging")
    return parser.parse_args()


def main() -> None:
    """Entrypoint used when the module is executed as a script."""
    args = parse_args()
    configure_logging(args.verbose)

    labels, features = load_dataset(args.data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=labels,
    )

    model = build_model(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
    )
    LOGGER.info("Training on %s samples (%s kept for evaluation)", len(X_train), len(X_test))
    model.fit(X_train, y_train)

    artifacts = evaluate(model, X_test, y_test)
    if args.cv_folds > 1:
        # Reuse the same hyper parameters for the CV evaluation.
        cv_model = build_model(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=args.random_state,
            n_jobs=args.n_jobs,
        )
        artifacts.cross_val = run_cross_validation(features, labels, folds=args.cv_folds, model=cv_model)

    save_model(model, args.model_path)
    save_report(args.report_path, artifacts)
    LOGGER.info("Training artifacts stored in %s", args.report_path)


if __name__ == "__main__":
    main()
