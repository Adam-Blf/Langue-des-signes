"""Interactive tool that captures labelled landmark samples for training.

The script mirrors the dataset structure expected by ``train_model.py``:
each line stores a letter label followed by 63 relative coordinates.  The
letter is chosen by pressing the key indicated in the overlay.
"""

from __future__ import annotations

import argparse
import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import cv2
import mediapipe as mp

LOGGER = logging.getLogger(__name__)
WINDOW_TITLE = "Hand Tracking Collector"

# The first landmark produced by Mediapipe corresponds to the wrist.
WRIST_INDEX = 0


@dataclass(frozen=True)
class LetterBinding:
    """Represents the mapping between a keyboard key and the target letter."""

    key: str
    letter: str


def configure_logging(verbose: bool) -> None:
    """Configure basic logging so the CLI remains informative."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(format="%(levelname)s: %(message)s", level=level)


def parse_args() -> argparse.Namespace:
    """Set up and parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Capture labelled Mediapipe hand landmarks for training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "data.csv",
        help="CSV file where samples are appended",
    )
    parser.add_argument("--camera-index", type=int, default=0, help="Index of the webcam to open")
    parser.add_argument("--min-detection", type=float, default=0.7, help="Mediapipe detection confidence threshold")
    parser.add_argument("--min-tracking", type=float, default=0.5, help="Mediapipe tracking confidence threshold")
    parser.add_argument(
        "--letters",
        nargs="+",
        default=["A", "B", "C", "D", "E", "F"],
        help="Letters to capture; they are mapped to keys in alphabetical order",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the destination CSV instead of appending to it",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    return parser.parse_args()


def build_bindings(letters: Sequence[str]) -> List[LetterBinding]:
    """Return keyboard bindings generated from *letters*."""
    bindings: List[LetterBinding] = []
    for index, letter in enumerate(letters):
        key = chr(ord("a") + index)
        bindings.append(LetterBinding(key=key, letter=letter.upper()))
    return bindings


def ensure_output_file(path: Path, overwrite: bool) -> None:
    """Create the parent directory if needed and optionally empty the file."""
    LOGGER.debug("Preparing output file at %s", path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite:
        path.write_text("", encoding="utf-8")


def flatten_landmarks(hand_landmarks) -> List[float]:
    """Return the 63 relative coordinates derived from *hand_landmarks*."""
    wrist = hand_landmarks.landmark[WRIST_INDEX]
    flattened: List[float] = []
    for landmark in hand_landmarks.landmark:
        flattened.extend(
            (
                landmark.x - wrist.x,
                landmark.y - wrist.y,
                landmark.z - wrist.z,
            )
        )
    return flattened


class SampleCollector:
    """High-level orchestrator that handles the webcam loop and CSV writing."""

    def __init__(
        self,
        *,
        bindings: Sequence[LetterBinding],
        output_path: Path,
        camera_index: int,
        min_detection: float,
        min_tracking: float,
    ) -> None:
        self.bindings = bindings
        self.output_path = output_path
        self.camera_index = camera_index
        self.min_detection = min_detection
        self.min_tracking = min_tracking

        self._hands = mp.solutions.hands.Hands(
            min_detection_confidence=self.min_detection,
            min_tracking_confidence=self.min_tracking,
        )
        self._drawer = mp.solutions.drawing_utils

        self._samples: List[Tuple[str, List[float]]] = []

    def _open_camera(self) -> cv2.VideoCapture:
        """Open the requested camera and raise if it cannot be accessed."""
        LOGGER.debug("Opening camera %s", self.camera_index)
        capture = cv2.VideoCapture(self.camera_index)
        if not capture.isOpened():
            raise RuntimeError(f"Cannot access camera index {self.camera_index}")
        return capture

    def _handle_key(self, key_code: int, landmarks: List[float]) -> None:
        """Store a sample when *key_code* matches one of the bindings."""
        for binding in self.bindings:
            if key_code == ord(binding.key):
                LOGGER.info("Captured sample for letter %s", binding.letter)
                self._samples.append((binding.letter, landmarks))
                break

    def _draw_overlay(self, frame) -> None:
        """Render helper text on top of the current frame."""
        instructions = [f"{binding.key.upper()} -> {binding.letter}" for binding in self.bindings]
        top_message = "Press a mapped key to record, ESC to quit"
        cv2.putText(frame, top_message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        for idx, message in enumerate(instructions, start=1):
            cv2.putText(
                frame,
                message,
                (10, 30 + idx * 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
        cv2.putText(
            frame,
            f"Samples: {len(self._samples)}",
            (10, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
        )

    def run(self) -> None:
        """Main capture loop: draw landmarks, record samples, and mirror feedback."""
        cap = self._open_camera()
        try:
            while True:
                success, frame = cap.read()
                if not success:
                    LOGGER.warning("Unable to read a frame from the camera")
                    break

                # Mirror the frame so gestures look natural.
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self._hands.process(rgb_frame)

                latest_landmarks: List[float] | None = None
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        self._drawer.draw_landmarks(
                            frame,
                            hand_landmarks,
                            mp.solutions.hands.HAND_CONNECTIONS,
                        )
                        latest_landmarks = flatten_landmarks(hand_landmarks)

                self._draw_overlay(frame)
                cv2.imshow(WINDOW_TITLE, frame)

                key_code = cv2.waitKey(1) & 0xFF
                if key_code == 27:
                    LOGGER.info("ESC pressed, closing capture loop")
                    break
                if latest_landmarks is not None:
                    self._handle_key(key_code, latest_landmarks)
        finally:
            cap.release()
            cv2.destroyAllWindows()

    def save(self) -> None:
        """Append the collected samples to the CSV file."""
        if not self._samples:
            LOGGER.info("No samples collected, skipping file write")
            return

        LOGGER.info("Writing %s new samples to %s", len(self._samples), self.output_path)
        with self.output_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            for letter, coords in self._samples:
                writer.writerow([letter] + coords)


def main() -> None:
    """Entrypoint called when executing the module as a script."""
    args = parse_args()
    configure_logging(args.verbose)

    bindings = build_bindings(args.letters)
    ensure_output_file(args.output, args.overwrite)
    collector = SampleCollector(
        bindings=bindings,
        output_path=args.output,
        camera_index=args.camera_index,
        min_detection=args.min_detection,
        min_tracking=args.min_tracking,
    )
    LOGGER.info("Ready to collect samples. Window: %s", WINDOW_TITLE)
    LOGGER.info(
        "Mappings: %s",
        ", ".join(f"{binding.key.upper()}->{binding.letter}" for binding in bindings),
    )
    collector.run()
    collector.save()


if __name__ == "__main__":
    main()
