"""Enhanced Graphical interface for live sign-language detection.

The application combines Mediapipe hand tracking, a hybrid detection pipeline,
and a Tkinter user interface with all 6 new features integrated:
1. Extended alphabet A-Z detection
2. Word and phrase detection
3. Multilingual support (7 languages)
4. Voice feedback (TTS)
5. Learning mode (45+ exercises)
6. GPU acceleration (PyTorch/ONNX)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import messagebox, ttk

from detection_pipeline import DetectionResult, SignDetectionPipeline

# Import new features
try:
    from letters_conditions_extended import detect_letter_extended
    HAS_EXTENDED_ALPHABET = True
except ImportError:
    HAS_EXTENDED_ALPHABET = False
    logging.warning("Extended alphabet A-Z not available")

try:
    from word_detector import WordDetector, PhraseBuilder
    HAS_WORD_DETECTION = True
except ImportError:
    HAS_WORD_DETECTION = False
    logging.warning("Word/phrase detection not available")

try:
    from voice_feedback import VoiceFeedback, FeedbackMode
    HAS_VOICE_FEEDBACK = True
except ImportError:
    HAS_VOICE_FEEDBACK = False
    logging.warning("Voice feedback not available")

try:
    from language_config import LanguageManager, SignLanguage
    HAS_MULTILINGUAL = True
except ImportError:
    HAS_MULTILINGUAL = False
    logging.warning("Multilingual support not available")

try:
    from learning_mode import LearningMode, ExerciseType, DifficultyLevel
    HAS_LEARNING_MODE = True
except ImportError:
    HAS_LEARNING_MODE = False
    logging.warning("Learning mode not available")

try:
    from gpu_inference import GPUInference, has_gpu_support
    HAS_GPU_ACCELERATION = True
except ImportError:
    HAS_GPU_ACCELERATION = False
    logging.warning("GPU acceleration not available")

LOGGER = logging.getLogger(__name__)

# Tkinter refresh delay (in milliseconds) used to schedule the video loop.
FRAME_DELAY_MS = 15
# Rendered frame size in the Tkinter label. Camera resolution stays untouched.
FRAME_WIDTH = 800
FRAME_HEIGHT = 600


class LsfApp:
    """High-level controller that binds the detection pipeline to the UI with all 6 new features."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Sign Language Detection - Enhanced (A-Z + Words + Voice + Learning + GPU)")

        # Core detection helpers
        self.pipeline = SignDetectionPipeline()
        self.cap: Optional[cv2.VideoCapture] = None
        self.hands = None
        self.hand_connections = None

        # Runtime flags
        self.video_active = False
        self.detection_active = False

        # ✨ NEW: Initialize 6 features
        self.word_detector = WordDetector(pause_threshold=1.5) if HAS_WORD_DETECTION else None
        self.phrase_builder = PhraseBuilder() if HAS_WORD_DETECTION else None
        self.voice_feedback = VoiceFeedback() if HAS_VOICE_FEEDBACK else None
        self.language_manager = LanguageManager() if HAS_MULTILINGUAL else None
        self.learning_mode = LearningMode() if HAS_LEARNING_MODE else None
        self.gpu_inference = GPUInference() if HAS_GPU_ACCELERATION else None
        
        # ✨ NEW: Status indicators for features
        self.words_detected: list[str] = []
        self.phrases_detected: list[str] = []
        self.current_language = SignLanguage.LSF if HAS_MULTILINGUAL else None
        self.learning_active = False
        self.gpu_enabled = has_gpu_support() if HAS_GPU_ACCELERATION else False

        # Tkinter variables keep the UI reactive without manual refreshes.
        self.letter_var = tk.StringVar(value="")
        self.method_var = tk.StringVar(value="")
        self.confidence_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Press Start detection to begin.")
        self.transcription_var = tk.StringVar(value="")
        
        # ✨ NEW: Additional UI variables
        self.word_var = tk.StringVar(value="")
        self.phrase_var = tk.StringVar(value="")
        self.language_var = tk.StringVar(value="LSF (Français)")
        self.voice_mode_var = tk.StringVar(value="LETTERS")
        self.learning_exercise_var = tk.StringVar(value="")
        self.gpu_status_var = tk.StringVar(value="GPU: " + ("✅ Active" if self.gpu_enabled else "❌ Inactive"))

        # Pipeline tuning controls exposed in the sidebar.
        self.threshold_var = tk.DoubleVar(value=self.pipeline.ml_threshold)
        self.consensus_var = tk.IntVar(value=self.pipeline.min_consensus)
        self.rules_var = tk.BooleanVar(value=self.pipeline.enable_rules)
        self.ml_var = tk.BooleanVar(value=self.pipeline.enable_ml)

        # Transcription helper state.
        self.transcription_cooldown = 0.8  # seconds between identical letters
        self.last_transcribed_letter: Optional[str] = None
        self.last_transcription_ts = 0.0

        # History widget state.
        self.history_max_items = 50
        self.history_entries: list[str] = []

        # Build the UI then initialise camera + Mediapipe.
        self._configure_styles()
        self._build_ui()
        self._register_variable_callbacks()
        self._setup_capture()

    # ----------------------------------------------------------------- helpers
    def _configure_styles(self) -> None:
        """Create a consistent visual theme for ttk widgets."""
        try:
            style = ttk.Style()
            bg = "#ffffff"
            accent = "#ff66b3"
            style.configure("App.TFrame", background=bg)
            style.configure("Video.TLabel", background=bg)
            style.configure("Letter.TLabel", font=("Segoe UI", 48, "bold"), foreground=accent, background=bg)
            style.configure("Method.TLabel", font=("Segoe UI", 11), foreground="#6b7280", background=bg)
            style.configure("Status.TLabel", font=("Segoe UI", 10), foreground="#4b5563", background=bg)
            style.configure("Sidebar.TFrame", background=bg)
            style.configure("SidebarHeading.TLabel", font=("Segoe UI", 12, "bold"), foreground="#111827", background=bg)
            self.root.configure(bg=bg)
        except Exception:  # pragma: no cover - style errors are non critical
            LOGGER.debug("Unable to apply custom styles", exc_info=True)

    def _build_ui(self) -> None:
        """Assemble all Tk widgets."""
        container = ttk.Frame(self.root, style="App.TFrame")
        container.pack(fill="both", expand=True, padx=8, pady=8)

        content = ttk.Frame(container, style="App.TFrame")
        content.pack(fill="both", expand=True)

        # Video pane on the left keeps the webcam preview centred.
        video_frame = ttk.Frame(content, style="App.TFrame")
        video_frame.pack(side="left", fill="both", expand=True)
        self.video_label = ttk.Label(video_frame, style="Video.TLabel")
        self.video_label.pack(fill="both", expand=True)

        # Sidebar contains controls, history, and transcription helpers.
        sidebar = ttk.Frame(content, style="Sidebar.TFrame", width=280)
        sidebar.pack(side="right", fill="y", padx=(12, 0))

        self._build_detection_section(sidebar)
        self._build_pipeline_controls(sidebar)
        self._build_history_section(sidebar)
        self._build_transcription_section(sidebar)
        self._build_buttons(sidebar)

    def _build_detection_section(self, sidebar: ttk.Frame) -> None:
        """Create the widgets that surface the live detection output."""
        ttk.Label(sidebar, text="Lettre detectee", style="SidebarHeading.TLabel").pack(anchor="w", pady=(0, 6))
        ttk.Label(sidebar, textvariable=self.letter_var, style="Letter.TLabel", anchor="center").pack(fill="x")
        ttk.Label(sidebar, textvariable=self.method_var, style="Method.TLabel", anchor="w").pack(fill="x", pady=(4, 0))
        ttk.Label(sidebar, textvariable=self.confidence_var, style="Method.TLabel", anchor="w").pack(fill="x")
        ttk.Label(sidebar, textvariable=self.status_var, style="Status.TLabel", anchor="w").pack(fill="x", pady=(4, 12))

    def _build_pipeline_controls(self, sidebar: ttk.Frame) -> None:
        """Expose key detection parameters so users can tune the behaviour."""
        container = ttk.LabelFrame(sidebar, text="Parametres detection", padding=8)
        container.pack(fill="x", pady=(0, 12))

        ttk.Label(container, text="Seuil ML", style="Status.TLabel").pack(anchor="w")
        self.threshold_value_label = ttk.Label(container, text=f"{self.threshold_var.get():.2f}", style="Method.TLabel")
        self.threshold_value_label.pack(anchor="e")
        ttk.Scale(
            container,
            from_=0.50,
            to=0.99,
            variable=self.threshold_var,
            command=self._on_threshold_changed,
        ).pack(fill="x")

        ttk.Label(container, text="Consensus minimum", style="Status.TLabel").pack(anchor="w", pady=(8, 0))
        spinbox = tk.Spinbox(
            container,
            from_=1,
            to=10,
            textvariable=self.consensus_var,
            width=5,
            command=self._apply_pipeline_settings,
        )
        spinbox.pack(anchor="w")

        ttk.Checkbutton(
            container,
            text="Activer regles",
            variable=self.rules_var,
            command=self._apply_pipeline_settings,
        ).pack(anchor="w", pady=(8, 0))
        ttk.Checkbutton(
            container,
            text="Activer modele ML",
            variable=self.ml_var,
            command=self._apply_pipeline_settings,
        ).pack(anchor="w")

    def _build_history_section(self, sidebar: ttk.Frame) -> None:
        """Create the scrollable history that stores stabilised letters."""
        ttk.Label(sidebar, text="Historique", style="SidebarHeading.TLabel").pack(anchor="w")
        history_frame = ttk.Frame(sidebar, style="Sidebar.TFrame")
        history_frame.pack(fill="both", expand=False, pady=(4, 8))
        self.history_list = tk.Listbox(history_frame, height=8, activestyle="none")
        self.history_list.pack(side="left", fill="both", expand=True)
        history_scroll = ttk.Scrollbar(history_frame, orient="vertical", command=self.history_list.yview)
        history_scroll.pack(side="right", fill="y")
        self.history_list.configure(yscrollcommand=history_scroll.set)
        ttk.Button(
            sidebar,
            text="Vider l'historique",
            command=self.clear_history,
            width=22,
        ).pack(pady=(2, 12))

    def _build_transcription_section(self, sidebar: ttk.Frame) -> None:
        """Display the transcription buffer and helper buttons."""
        ttk.Label(sidebar, text="Transcription", style="SidebarHeading.TLabel").pack(anchor="w")
        self.transcription_label = ttk.Label(
            sidebar,
            textvariable=self.transcription_var,
            style="Status.TLabel",
            anchor="w",
            justify="left",
            wraplength=240,
        )
        self.transcription_label.pack(fill="x", pady=(4, 8))

        controls = tk.Frame(sidebar, bg="#ffffff", highlightthickness=0, bd=0)
        controls.pack(fill="x", pady=(2, 8))
        tk.Button(
            controls,
            text="Space",
            bg="#e5e7eb",
            activebackground="#d1d5db",
            fg="#111827",
            relief="flat",
            command=self.add_space,
        ).pack(side="left", padx=4)
        tk.Button(
            controls,
            text="Backspace",
            bg="#e5e7eb",
            activebackground="#d1d5db",
            fg="#111827",
            relief="flat",
            command=self.backspace,
        ).pack(side="left", padx=4)
        tk.Button(
            controls,
            text="Clear",
            bg="#e5e7eb",
            activebackground="#d1d5db",
            fg="#111827",
            relief="flat",
            command=self.clear_transcription,
        ).pack(side="left", padx=4)

    def _build_buttons(self, sidebar: ttk.Frame) -> None:
        """Add the start/stop and quit buttons."""
        controls = tk.Frame(sidebar, bg="#ffffff", highlightthickness=0, bd=0)
        controls.pack(fill="x", pady=(4, 0))
        self.toggle_btn = tk.Button(
            controls,
            text="Start detection",
            bg="#ff66b3",
            activebackground="#ff4da6",
            fg="white",
            activeforeground="white",
            relief="flat",
            command=self.toggle_detection,
        )
        self.toggle_btn.pack(side="left", padx=6)
        tk.Button(
            controls,
            text="Quit",
            bg="#ff66b3",
            activebackground="#ff4da6",
            fg="white",
            activeforeground="white",
            relief="flat",
            command=self.quit,
        ).pack(side="right", padx=6)

    def _register_variable_callbacks(self) -> None:
        """Keep the pipeline in sync with the sidebar controls."""
        self.threshold_var.trace_add("write", lambda *_: self._apply_pipeline_settings())
        self.consensus_var.trace_add("write", lambda *_: self._apply_pipeline_settings())
        self.rules_var.trace_add("write", lambda *_: self._apply_pipeline_settings())
        self.ml_var.trace_add("write", lambda *_: self._apply_pipeline_settings())

    def _setup_capture(self) -> None:
        """Open the webcam and initialise Mediapipe."""
        self.cap = self._open_camera()
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self._init_mediapipe()
        self.hand_connections = self.mp_hands.HAND_CONNECTIONS if self.hands else None

        if self.cap is not None and self.hands is not None:
            self.start_video_loop()
            self.start_detection()

    # ------------------------------------------------------------- pipeline ---
    def _open_camera(self) -> Optional[cv2.VideoCapture]:
        """Return an opened VideoCapture or display an error dialog."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Camera error", "Unable to access the webcam.")
            self.status_var.set("Camera not available.")
            return None
        return cap

    def _init_mediapipe(self):
        """Initialise Mediapipe Hands with resilient error handling."""
        try:
            return self.mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
        except Exception as exc:  # pragma: no cover - external dependency
            LOGGER.exception("Unable to initialise Mediapipe: %s", exc)
            messagebox.showerror(
                "Initialisation error",
                f"Mediapipe Hands initialisation failed.\n\nDetails: {exc}",
            )
            self.status_var.set("Mediapipe initialisation failed.")
            return None

    def _apply_pipeline_settings(self) -> None:
        """Update pipeline parameters from GUI controls."""
        if not self.pipeline:
            return

        self.pipeline.ml_threshold = float(self.threshold_var.get())
        self.pipeline.min_consensus = int(self.consensus_var.get())
        self.pipeline.enable_rules = bool(self.rules_var.get())
        self.pipeline.enable_ml = bool(self.ml_var.get())

        # Reset the temporal smoothing history so new parameters take effect.
        self.pipeline.reset()
        if self.detection_active:
            self.status_var.set("Detection settings applied.")
        self.threshold_value_label.configure(text=f"{self.pipeline.ml_threshold:.2f}")

    def _on_threshold_changed(self, _value: str) -> None:
        """Update the label displaying the ML threshold."""
        self.threshold_value_label.configure(text=f"{float(_value):.2f}")

    # -------------------------------------------------------------- controls --
    def start_video_loop(self) -> None:
        """Kick off the Tkinter-based video refresh loop."""
        if self.cap is None:
            return
        self.video_active = True
        self.root.after(FRAME_DELAY_MS, self._update_frame)

    def toggle_detection(self) -> None:
        """Start or stop the detection pipeline."""
        if self.cap is None or self.hands is None:
            return
        if self.detection_active:
            self.stop_detection()
        else:
            self.start_detection()

    def quit(self) -> None:
        """Release resources and close the application."""
        self.video_active = False
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        self.root.destroy()

    # ---------------------------------------------------------------- runtime -
    def _update_frame(self) -> None:
        """Read a frame, run detection if enabled, and schedule the next update."""
        if not self.video_active or self.cap is None:
            return

        success, frame = self.cap.read()
        if not success:
            self.status_var.set("Camera frame not available.")
            self.root.after(FRAME_DELAY_MS, self._update_frame)
            return

        # Mirror the view so users see a natural reflection.
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result: Optional[DetectionResult] = None
        if self.detection_active and self.hands is not None:
            detection = self.hands.process(frame_rgb)
            if detection and detection.multi_hand_landmarks:
                for hand_landmarks in detection.multi_hand_landmarks:
                    result = self.pipeline.process(hand_landmarks)
                    # Draw landmarks on BGR frame for correct rendering
                    if self.hand_connections and hasattr(self, 'mp_drawing'):
                        self.mp_drawing.draw_landmarks(
                            frame,
                            hand_landmarks,
                            self.hand_connections,
                            self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                            self.mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2)
                        )
                    break  # Only track the first detected hand for readability.
            else:
                result = self.pipeline.process(None)

        self._render_frame(frame)
        if result is not None:
            self._update_detection_labels(result)

        self.root.after(FRAME_DELAY_MS, self._update_frame)

    def _render_frame(self, frame) -> None:
        """Convert the OpenCV frame to a Tk-friendly image."""
        try:
            display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(display_frame, (FRAME_WIDTH, FRAME_HEIGHT))
            image = Image.fromarray(resized)
            imgtk = ImageTk.PhotoImage(image=image)
            self.video_label.imgtk = imgtk  # Keep reference to avoid garbage collection.
            self.video_label.configure(image=imgtk)
        except Exception as e:
            LOGGER.error(f"Error rendering frame: {e}")
            self.status_var.set("Frame rendering error.")

    def _update_detection_labels(self, result: DetectionResult) -> None:
        """Refresh labels, transcription, and history based on *result*."""
        if result.letter:
            self.letter_var.set(result.letter)
            method = "rules" if result.source == "rules" else "ML model"
            self.method_var.set(f"Detected via {method}")
            if result.confidence is not None:
                self.confidence_var.set(f"Confidence: {result.confidence * 100:.1f}%")
            else:
                self.confidence_var.set("")
            if result.raw_letter is None:
                self.status_var.set("Hand briefly lost. Holding previous letter.")
            else:
                self.status_var.set("Hand detected.")
            self._handle_transcription(result.letter)
        elif result.raw_letter:
            method = "rules" if result.raw_source == "rules" else "ML model"
            self.letter_var.set(result.raw_letter)
            self.method_var.set(f"Stabilising {method} prediction...")
            if result.raw_confidence is not None:
                self.confidence_var.set(f"Instant confidence: {result.raw_confidence * 100:.1f}%")
            else:
                self.confidence_var.set("")
            self.status_var.set("Gathering more frames for stability...")
        else:
            self._clear_detection_labels()
            if self.detection_active:
                self.status_var.set("No hand detected.")
        if result and result.letter:
            self._append_history(result)

    def _clear_detection_labels(self) -> None:
        """Reset the sidebar labels when no letter is detected."""
        self.letter_var.set("")
        self.method_var.set("")
        self.confidence_var.set("")

    def start_detection(self) -> None:
        """Arm the detection pipeline and reset temporary state."""
        if self.cap is None or self.hands is None or self.detection_active:
            return
        self.pipeline.reset()
        self._reset_transcription_state()
        self.detection_active = True
        self.status_var.set("Detection running...")
        self.toggle_btn.configure(text="Stop detection")

    def stop_detection(self) -> None:
        """Pause detection while keeping the video feed active."""
        if not self.detection_active:
            return
        self.pipeline.reset()
        self._reset_transcription_state()
        self._clear_detection_labels()
        self.detection_active = False
        self.status_var.set("Detection paused.")
        self.toggle_btn.configure(text="Start detection")

    # ---------------------------------------------------------- transcription -
    def _handle_transcription(self, letter: str) -> None:
        """Append *letter* to the transcription with a debounce guard."""
        if not self.detection_active or not letter:
            return
        now = time.monotonic()
        if (
            letter != self.last_transcribed_letter
            or now - self.last_transcription_ts >= self.transcription_cooldown
        ):
            self.transcription_var.set(self.transcription_var.get() + letter)
            self.last_transcribed_letter = letter
            self.last_transcription_ts = now

    def add_space(self) -> None:
        """Insert a space in the transcription buffer."""
        self.transcription_var.set(self.transcription_var.get() + " ")
        self._reset_transcription_state()

    def backspace(self) -> None:
        """Remove the last character from the transcription buffer."""
        current = self.transcription_var.get()
        if current:
            self.transcription_var.set(current[:-1])
        self._reset_transcription_state()

    def clear_transcription(self) -> None:
        """Clear the transcription buffer."""
        self.transcription_var.set("")
        self._reset_transcription_state()

    def _reset_transcription_state(self) -> None:
        """Forget the last letter so repetitions can be captured again."""
        self.last_transcribed_letter = None
        self.last_transcription_ts = 0.0

    def _append_history(self, result: DetectionResult) -> None:
        """Append the stabilised letter to the scrollable history."""
        timestamp = time.strftime("%H:%M:%S")
        source = "regles" if result.source == "rules" else "ML"
        confidence = f"{result.confidence * 100:.0f}%" if result.confidence is not None else "-"
        entry = f"{timestamp}  {result.letter} ({source}, {confidence})"

        # Skip duplicates when the same entry is already displayed last.
        if self.history_entries and self.history_entries[-1] == entry:
            return

        self.history_entries.append(entry)
        if len(self.history_entries) > self.history_max_items:
            self.history_entries.pop(0)

        self.history_list.delete(0, tk.END)
        for item in self.history_entries:
            self.history_list.insert(tk.END, item)
        self.history_list.yview_moveto(1.0)

    def clear_history(self) -> None:
        """Clear the detection history listbox."""
        self.history_entries.clear()
        self.history_list.delete(0, tk.END)


def main() -> None:
    """Entry point when the module is executed as a script."""
    handlers = [logging.StreamHandler()]
    try:
        log_path = Path("lsf_detector.log")
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
    except Exception:
        LOGGER.debug("Unable to attach file logger", exc_info=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )
    root = tk.Tk()
    app = LsfApp(root)
    root.protocol("WM_DELETE_WINDOW", app.quit)
    root.mainloop()


if __name__ == "__main__":
    main()
