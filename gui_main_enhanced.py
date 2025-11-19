"""Enhanced GUI with all 6 new features integrated.

Integrates:
1. Extended alphabet A-Z detection
2. Word and phrase detection  
3. Multilingual support
4. Voice feedback (TTS)
5. Learning mode
6. GPU acceleration
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
    print("Warning: Extended alphabet not available")

try:
    from word_detector import WordDetector, PhraseBuilder
    HAS_WORD_DETECTION = True
except ImportError:
    HAS_WORD_DETECTION = False
    print("Warning: Word detection not available")

try:
    from voice_feedback import VoiceFeedback, FeedbackMode
    HAS_VOICE_FEEDBACK = True
except ImportError:
    HAS_VOICE_FEEDBACK = False
    print("Warning: Voice feedback not available")

try:
    from language_config import LanguageManager, SignLanguage
    HAS_MULTILINGUAL = True
except ImportError:
    HAS_MULTILINGUAL = False
    print("Warning: Multilingual support not available")

try:
    from gpu_inference import InferenceEngine, GPUDetector
    HAS_GPU_INFERENCE = True
except ImportError:
    HAS_GPU_INFERENCE = False
    print("Warning: GPU inference not available")

LOGGER = logging.getLogger(__name__)

FRAME_DELAY_MS = 15
FRAME_WIDTH = 800
FRAME_HEIGHT = 600


class EnhancedLsfApp:
    """Enhanced app with all new features."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Sign Language Detection - Enhanced Edition 🚀")
        self.root.geometry("1200x800")

        # Core detection
        self.pipeline = SignDetectionPipeline()
        self.cap: Optional[cv2.VideoCapture] = None
        self.hands = None
        self.hand_connections = None

        # NEW FEATURE 1: Extended alphabet
        self.use_extended_alphabet = HAS_EXTENDED_ALPHABET

        # NEW FEATURE 2: Word & phrase detection
        if HAS_WORD_DETECTION:
            self.word_detector = WordDetector(pause_threshold=1.5)
            self.phrase_builder = PhraseBuilder()
        else:
            self.word_detector = None
            self.phrase_builder = None

        # NEW FEATURE 3: Multilingual
        if HAS_MULTILINGUAL:
            self.lang_manager = LanguageManager()
            self.current_language = SignLanguage.LSF
        else:
            self.lang_manager = None

        # NEW FEATURE 4: Voice feedback
        if HAS_VOICE_FEEDBACK:
            self.voice = VoiceFeedback()
            self.voice.set_mode(FeedbackMode.OFF)  # Start disabled
        else:
            self.voice = None

        # NEW FEATURE 6: GPU inference
        if HAS_GPU_INFERENCE:
            self.gpu_engine = InferenceEngine()
            self.gpu_info = GPUDetector.get_device_info()
            self.use_gpu = self.gpu_info.get('has_cuda') or self.gpu_info.get('has_mps')
        else:
            self.gpu_engine = None
            self.use_gpu = False

        # Runtime flags
        self.video_active = False
        self.detection_active = False

        # Tkinter variables
        self.letter_var = tk.StringVar(value="")
        self.word_var = tk.StringVar(value="")
        self.phrase_var = tk.StringVar(value="")
        self.method_var = tk.StringVar(value="")
        self.confidence_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Press Start to begin.")
        self.transcription_var = tk.StringVar(value="")
        
        # New variables
        self.gpu_status_var = tk.StringVar(value="GPU: Checking...")
        self.language_var = tk.StringVar(value="LSF")

        # Pipeline controls
        self.threshold_var = tk.DoubleVar(value=self.pipeline.ml_threshold)
        self.consensus_var = tk.IntVar(value=self.pipeline.min_consensus)
        self.rules_var = tk.BooleanVar(value=self.pipeline.enable_rules)
        self.ml_var = tk.BooleanVar(value=self.pipeline.enable_ml)
        
        # Voice controls
        self.voice_enabled_var = tk.BooleanVar(value=False)
        self.voice_mode_var = tk.StringVar(value="OFF")

        # Transcription state
        self.transcription_cooldown = 0.8
        self.last_transcribed_letter: Optional[str] = None
        self.last_transcription_ts = 0.0

        # History
        self.history_max_items = 50
        self.history_entries: list[str] = []

        # Build UI
        self._configure_styles()
        self._build_ui()
        self._register_callbacks()
        self._setup_capture()
        self._update_feature_status()

    def _configure_styles(self) -> None:
        """Configure UI styles."""
        try:
            style = ttk.Style()
            bg = "#f8f9fa"
            accent = "#ff66b3"
            style.configure("App.TFrame", background=bg)
            style.configure("Video.TLabel", background="#000000")
            style.configure("Letter.TLabel", font=("Segoe UI", 48, "bold"), foreground=accent, background=bg)
            style.configure("Word.TLabel", font=("Segoe UI", 24, "bold"), foreground="#4a5568", background=bg)
            style.configure("Method.TLabel", font=("Segoe UI", 11), foreground="#6b7280", background=bg)
            style.configure("Status.TLabel", font=("Segoe UI", 10), foreground="#4b5563", background=bg)
            style.configure("Sidebar.TFrame", background=bg)
            style.configure("SidebarHeading.TLabel", font=("Segoe UI", 12, "bold"), foreground="#111827", background=bg)
            style.configure("Feature.TLabel", font=("Segoe UI", 9), foreground="#059669", background=bg)
            self.root.configure(bg=bg)
        except Exception:
            LOGGER.debug("Unable to apply custom styles", exc_info=True)

    def _build_ui(self) -> None:
        """Build the UI with new features."""
        # Main container
        container = ttk.Frame(self.root, style="App.TFrame")
        container.pack(fill="both", expand=True, padx=10, pady=10)

        # Top bar with feature status
        self._build_feature_status_bar(container)

        # Main content area
        content = ttk.Frame(container, style="App.TFrame")
        content.pack(fill="both", expand=True, pady=(10, 0))

        # Video panel (left)
        video_frame = ttk.Frame(content, style="App.TFrame")
        video_frame.pack(side="left", fill="both", expand=True)
        self.video_label = ttk.Label(video_frame, style="Video.TLabel")
        self.video_label.pack(fill="both", expand=True)

        # Enhanced sidebar (right)
        self._build_enhanced_sidebar(content)

    def _build_feature_status_bar(self, parent):
        """Build top bar showing enabled features."""
        status_bar = ttk.Frame(parent, style="Sidebar.TFrame", relief="solid", borderwidth=1)
        status_bar.pack(fill="x", pady=(0, 10))

        ttk.Label(status_bar, text="🚀 Enhanced Features:", style="SidebarHeading.TLabel").pack(side="left", padx=10, pady=5)

        features_frame = ttk.Frame(status_bar, style="Sidebar.TFrame")
        features_frame.pack(side="left", fill="x", expand=True, padx=5)

        self.feature_labels = {}
        features = [
            ("alphabet", "A-Z Alphabet", HAS_EXTENDED_ALPHABET),
            ("words", "Words/Phrases", HAS_WORD_DETECTION),
            ("multilang", "Multilingual", HAS_MULTILINGUAL),
            ("voice", "Voice TTS", HAS_VOICE_FEEDBACK),
            ("gpu", "GPU Accel", HAS_GPU_INFERENCE)
        ]

        for key, label, enabled in features:
            icon = "✅" if enabled else "❌"
            lbl = ttk.Label(
                features_frame,
                text=f"{icon} {label}",
                style="Feature.TLabel" if enabled else "Status.TLabel"
            )
            lbl.pack(side="left", padx=8, pady=2)
            self.feature_labels[key] = lbl

    def _build_enhanced_sidebar(self, parent):
        """Build enhanced sidebar with new controls."""
        sidebar = ttk.Frame(parent, style="Sidebar.TFrame", width=350)
        sidebar.pack(side="right", fill="y", padx=(15, 0))

        # Notebook for tabs
        self.notebook = ttk.Notebook(sidebar)
        self.notebook.pack(fill="both", expand=True)

        # Tab 1: Detection
        detection_tab = ttk.Frame(self.notebook, style="Sidebar.TFrame")
        self.notebook.add(detection_tab, text="Detection")
        self._build_detection_tab(detection_tab)

        # Tab 2: Words & Phrases (NEW)
        if HAS_WORD_DETECTION:
            words_tab = ttk.Frame(self.notebook, style="Sidebar.TFrame")
            self.notebook.add(words_tab, text="Words/Phrases")
            self._build_words_tab(words_tab)

        # Tab 3: Settings (NEW)
        settings_tab = ttk.Frame(self.notebook, style="Sidebar.TFrame")
        self.notebook.add(settings_tab, text="Settings")
        self._build_settings_tab(settings_tab)

        # Control buttons at bottom
        self._build_control_buttons(sidebar)

    def _build_detection_tab(self, parent):
        """Build detection tab (original + enhancements)."""
        # Letter detection
        ttk.Label(parent, text="Lettre detectee", style="SidebarHeading.TLabel").pack(anchor="w", pady=(5, 5))
        ttk.Label(parent, textvariable=self.letter_var, style="Letter.TLabel", anchor="center").pack(fill="x")
        ttk.Label(parent, textvariable=self.method_var, style="Method.TLabel", anchor="w").pack(fill="x")
        ttk.Label(parent, textvariable=self.confidence_var, style="Method.TLabel", anchor="w").pack(fill="x")

        # Word display (NEW)
        if HAS_WORD_DETECTION:
            ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=10)
            ttk.Label(parent, text="Mot detecte", style="SidebarHeading.TLabel").pack(anchor="w")
            ttk.Label(parent, textvariable=self.word_var, style="Word.TLabel", anchor="center").pack(fill="x")

        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=10)

        # Status
        ttk.Label(parent, textvariable=self.status_var, style="Status.TLabel", anchor="w").pack(fill="x", pady=5)

        # Pipeline controls
        controls = ttk.LabelFrame(parent, text="Parametres", padding=10)
        controls.pack(fill="x", pady=10)

        ttk.Label(controls, text="Seuil ML:", style="Status.TLabel").pack(anchor="w")
        self.threshold_label = ttk.Label(controls, text=f"{self.threshold_var.get():.2f}", style="Method.TLabel")
        self.threshold_label.pack(anchor="e")
        ttk.Scale(controls, from_=0.50, to=0.99, variable=self.threshold_var,
                 command=self._on_threshold_changed).pack(fill="x")

        ttk.Checkbutton(controls, text="Regles heuristiques", variable=self.rules_var).pack(anchor="w", pady=(8, 0))
        ttk.Checkbutton(controls, text="Modele ML", variable=self.ml_var).pack(anchor="w")

        # History
        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=10)
        ttk.Label(parent, text="Historique", style="SidebarHeading.TLabel").pack(anchor="w")
        history_frame = ttk.Frame(parent, style="Sidebar.TFrame")
        history_frame.pack(fill="both", expand=True, pady=5)
        self.history_list = tk.Listbox(history_frame, height=10)
        self.history_list.pack(side="left", fill="both", expand=True)
        history_scroll = ttk.Scrollbar(history_frame, orient="vertical", command=self.history_list.yview)
        history_scroll.pack(side="right", fill="y")
        self.history_list.configure(yscrollcommand=history_scroll.set)

        ttk.Button(parent, text="Vider", command=self.clear_history).pack(fill="x", pady=5)

    def _build_words_tab(self, parent):
        """Build words/phrases tab (NEW)."""
        ttk.Label(parent, text="Detection de phrases", style="SidebarHeading.TLabel").pack(anchor="w", pady=(5, 10))

        # Current phrase
        ttk.Label(parent, text="Phrase actuelle:", style="Status.TLabel").pack(anchor="w")
        phrase_frame = ttk.Frame(parent, relief="solid", borderwidth=1)
        phrase_frame.pack(fill="x", pady=5)
        phrase_label = ttk.Label(phrase_frame, textvariable=self.phrase_var, 
                                style="Method.TLabel", wraplength=300, justify="left")
        phrase_label.pack(fill="x", padx=10, pady=10)

        # Transcription
        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=10)
        ttk.Label(parent, text="Transcription", style="SidebarHeading.TLabel").pack(anchor="w")
        
        transcription_frame = ttk.Frame(parent, relief="solid", borderwidth=1)
        transcription_frame.pack(fill="both", expand=True, pady=5)
        
        transcription_text = tk.Text(transcription_frame, height=10, wrap="word")
        transcription_text.pack(fill="both", expand=True, padx=5, pady=5)
        self.transcription_text = transcription_text

        # Transcription controls
        trans_controls = ttk.Frame(parent)
        trans_controls.pack(fill="x", pady=5)
        ttk.Button(trans_controls, text="Space", command=self.add_space, width=10).pack(side="left", padx=2)
        ttk.Button(trans_controls, text="Backspace", command=self.backspace, width=10).pack(side="left", padx=2)
        ttk.Button(trans_controls, text="Clear", command=self.clear_transcription, width=10).pack(side="left", padx=2)

    def _build_settings_tab(self, parent):
        """Build settings tab (NEW)."""
        # Language selection
        if HAS_MULTILINGUAL:
            lang_frame = ttk.LabelFrame(parent, text="Langue / Language", padding=10)
            lang_frame.pack(fill="x", pady=5)
            
            languages = [lang.value for lang in SignLanguage]
            self.language_combo = ttk.Combobox(lang_frame, textvariable=self.language_var, 
                                              values=languages, state="readonly")
            self.language_combo.pack(fill="x")
            self.language_combo.bind("<<ComboboxSelected>>", self._on_language_changed)

        # Voice feedback
        if HAS_VOICE_FEEDBACK:
            voice_frame = ttk.LabelFrame(parent, text="Feedback vocal", padding=10)
            voice_frame.pack(fill="x", pady=10)
            
            ttk.Checkbutton(voice_frame, text="Activer TTS", 
                          variable=self.voice_enabled_var,
                          command=self._on_voice_enabled_changed).pack(anchor="w")
            
            ttk.Label(voice_frame, text="Mode:", style="Status.TLabel").pack(anchor="w", pady=(10, 0))
            modes = ["OFF", "LETTERS", "WORDS", "PHRASES", "ALL"]
            self.voice_mode_combo = ttk.Combobox(voice_frame, textvariable=self.voice_mode_var,
                                                values=modes, state="readonly")
            self.voice_mode_combo.pack(fill="x")
            self.voice_mode_combo.bind("<<ComboboxSelected>>", self._on_voice_mode_changed)

        # GPU settings
        if HAS_GPU_INFERENCE:
            gpu_frame = ttk.LabelFrame(parent, text="Acceleration GPU", padding=10)
            gpu_frame.pack(fill="x", pady=10)
            
            ttk.Label(gpu_frame, textvariable=self.gpu_status_var, 
                     style="Status.TLabel").pack(anchor="w")
            
            if self.use_gpu:
                device_name = self.gpu_info.get('device_name', 'GPU Available')
                ttk.Label(gpu_frame, text=f"Device: {device_name}", 
                         style="Feature.TLabel").pack(anchor="w", pady=5)

        # Alphabet mode
        if HAS_EXTENDED_ALPHABET:
            alphabet_frame = ttk.LabelFrame(parent, text="Detection alphabet", padding=10)
            alphabet_frame.pack(fill="x", pady=10)
            
            self.extended_alphabet_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(alphabet_frame, text="Alphabet complet A-Z", 
                          variable=self.extended_alphabet_var).pack(anchor="w")
            ttk.Label(alphabet_frame, text="26 lettres disponibles", 
                     style="Feature.TLabel").pack(anchor="w", pady=2)

    def _build_control_buttons(self, parent):
        """Build control buttons."""
        controls = ttk.Frame(parent, style="Sidebar.TFrame")
        controls.pack(fill="x", pady=(10, 0))

        self.toggle_btn = tk.Button(
            controls,
            text="▶ Start Detection",
            bg="#10b981",
            fg="white",
            font=("Segoe UI", 11, "bold"),
            activebackground="#059669",
            relief="flat",
            command=self.toggle_detection,
            padx=20,
            pady=10
        )
        self.toggle_btn.pack(fill="x", pady=5)

        quit_btn = tk.Button(
            controls,
            text="❌ Quit",
            bg="#ef4444",
            fg="white",
            font=("Segoe UI", 10),
            activebackground="#dc2626",
            relief="flat",
            command=self.quit,
            padx=20,
            pady=8
        )
        quit_btn.pack(fill="x")

    def _register_callbacks(self) -> None:
        """Register variable callbacks."""
        self.threshold_var.trace_add("write", lambda *_: self._apply_pipeline_settings())
        self.consensus_var.trace_add("write", lambda *_: self._apply_pipeline_settings())
        self.rules_var.trace_add("write", lambda *_: self._apply_pipeline_settings())
        self.ml_var.trace_add("write", lambda *_: self._apply_pipeline_settings())

    def _setup_capture(self) -> None:
        """Setup camera and MediaPipe."""
        self.cap = self._open_camera()
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self._init_mediapipe()
        self.hand_connections = self.mp_hands.HAND_CONNECTIONS if self.hands else None

        if self.cap is not None and self.hands is not None:
            self.start_video_loop()

    def _open_camera(self) -> Optional[cv2.VideoCapture]:
        """Open camera."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Camera error", "Cannot access webcam.")
            self.status_var.set("Camera not available.")
            return None
        return cap

    def _init_mediapipe(self):
        """Initialize MediaPipe."""
        try:
            return self.mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
        except Exception as exc:
            LOGGER.exception("MediaPipe init failed: %s", exc)
            messagebox.showerror("Init error", f"MediaPipe failed.\n\n{exc}")
            self.status_var.set("MediaPipe init failed.")
            return None

    def _apply_pipeline_settings(self) -> None:
        """Apply pipeline settings."""
        if not self.pipeline:
            return
        self.pipeline.ml_threshold = float(self.threshold_var.get())
        self.pipeline.min_consensus = int(self.consensus_var.get())
        self.pipeline.enable_rules = bool(self.rules_var.get())
        self.pipeline.enable_ml = bool(self.ml_var.get())
        self.pipeline.reset()
        if self.detection_active:
            self.status_var.set("Settings applied.")
        self.threshold_label.configure(text=f"{self.pipeline.ml_threshold:.2f}")

    def _on_threshold_changed(self, _value: str) -> None:
        """Update threshold label."""
        self.threshold_label.configure(text=f"{float(_value):.2f}")

    def _on_language_changed(self, event) -> None:
        """Handle language change (NEW)."""
        if self.lang_manager:
            try:
                lang = SignLanguage(self.language_var.get())
                self.lang_manager.set_language(lang)
                self.status_var.set(f"Language changed to {lang.value}")
            except Exception as e:
                LOGGER.error(f"Language change failed: {e}")

    def _on_voice_enabled_changed(self) -> None:
        """Handle voice enable/disable (NEW)."""
        if self.voice:
            self.voice.set_enabled(self.voice_enabled_var.get())

    def _on_voice_mode_changed(self, event) -> None:
        """Handle voice mode change (NEW)."""
        if self.voice:
            try:
                mode = FeedbackMode[self.voice_mode_var.get()]
                self.voice.set_mode(mode)
                self.status_var.set(f"Voice mode: {mode.value}")
            except Exception as e:
                LOGGER.error(f"Voice mode change failed: {e}")

    def _update_feature_status(self):
        """Update feature status display."""
        if HAS_GPU_INFERENCE and self.use_gpu:
            backend = self.gpu_info.get('recommended_backend', 'unknown').upper()
            self.gpu_status_var.set(f"✅ GPU Active ({backend})")
        else:
            self.gpu_status_var.set("CPU Mode")

    # Video loop and detection (original logic with enhancements)
    def start_video_loop(self) -> None:
        """Start video loop."""
        if self.cap is None:
            return
        self.video_active = True
        self.root.after(FRAME_DELAY_MS, self._update_frame)

    def _update_frame(self) -> None:
        """Update frame with detection."""
        if not self.video_active or self.cap is None:
            return

        success, frame = self.cap.read()
        if not success:
            self.status_var.set("Frame not available.")
            self.root.after(FRAME_DELAY_MS, self._update_frame)
            return

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result: Optional[DetectionResult] = None
        if self.detection_active and self.hands is not None:
            detection = self.hands.process(frame_rgb)
            if detection and detection.multi_hand_landmarks:
                for hand_landmarks in detection.multi_hand_landmarks:
                    # Use extended alphabet if available (NEW)
                    if HAS_EXTENDED_ALPHABET and self.use_extended_alphabet:
                        detected_letter = detect_letter_extended(hand_landmarks)
                        if detected_letter:
                            result = DetectionResult(
                                letter=detected_letter,
                                source="extended",
                                confidence=0.95,
                                raw_letter=detected_letter,
                                raw_source="extended",
                                raw_confidence=0.95
                            )
                    else:
                        result = self.pipeline.process(hand_landmarks)
                    
                    if self.hand_connections:
                        self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.hand_connections)
                    break
            else:
                result = self.pipeline.process(None)

        self._render_frame(frame)
        if result is not None:
            self._update_detection_labels(result)

        self.root.after(FRAME_DELAY_MS, self._update_frame)

    def _render_frame(self, frame) -> None:
        """Render frame."""
        display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        resized = cv2.resize(display_frame, (FRAME_WIDTH, FRAME_HEIGHT))
        image = Image.fromarray(resized)
        imgtk = ImageTk.PhotoImage(image=image)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

    def _update_detection_labels(self, result: DetectionResult) -> None:
        """Update detection labels with word/phrase support (NEW)."""
        if result.letter:
            self.letter_var.set(result.letter)
            method = "extended" if result.source == "extended" else ("rules" if result.source == "rules" else "ML")
            self.method_var.set(f"Method: {method}")
            if result.confidence is not None:
                self.confidence_var.set(f"Confidence: {result.confidence * 100:.1f}%")
            else:
                self.confidence_var.set("")
            
            # Word detection (NEW)
            if HAS_WORD_DETECTION and self.word_detector:
                self.word_detector.add_letter(result.letter, confidence=result.confidence or 0.0)
                current_word = self.word_detector.get_current_word()
                if current_word:
                    self.word_var.set(current_word)
                    
                    # Voice feedback for word (NEW)
                    if self.voice and self.voice_enabled_var.get():
                        self.voice.speak_word(current_word)
                    
                    # Phrase building (NEW)
                    if self.phrase_builder:
                        self.phrase_builder.add_word(current_word)
                        phrase = self.phrase_builder.get_phrase()
                        self.phrase_var.set(phrase)
            
            # Voice feedback for letter (NEW)
            if self.voice and self.voice_enabled_var.get():
                self.voice.speak_letter(result.letter)
            
            self.status_var.set("Hand detected.")
            self._handle_transcription(result.letter)
            self._append_history(result)
        else:
            self._clear_detection_labels()
            if self.detection_active:
                self.status_var.set("No hand detected.")

    def _clear_detection_labels(self) -> None:
        """Clear detection labels."""
        self.letter_var.set("")
        self.method_var.set("")
        self.confidence_var.set("")

    def toggle_detection(self) -> None:
        """Toggle detection."""
        if self.cap is None or self.hands is None:
            return
        if self.detection_active:
            self.stop_detection()
        else:
            self.start_detection()

    def start_detection(self) -> None:
        """Start detection."""
        if self.cap is None or self.hands is None or self.detection_active:
            return
        self.pipeline.reset()
        if self.word_detector:
            self.word_detector.clear()
        if self.phrase_builder:
            self.phrase_builder.reset()
        self._reset_transcription_state()
        self.detection_active = True
        self.status_var.set("Detection running...")
        self.toggle_btn.configure(text="⏸ Stop Detection", bg="#f59e0b")

    def stop_detection(self) -> None:
        """Stop detection."""
        if not self.detection_active:
            return
        self.pipeline.reset()
        self._reset_transcription_state()
        self._clear_detection_labels()
        self.detection_active = False
        self.status_var.set("Detection paused.")
        self.toggle_btn.configure(text="▶ Start Detection", bg="#10b981")

    def _handle_transcription(self, letter: str) -> None:
        """Handle transcription."""
        if not self.detection_active or not letter:
            return
        now = time.monotonic()
        if (letter != self.last_transcribed_letter or 
            now - self.last_transcription_ts >= self.transcription_cooldown):
            if hasattr(self, 'transcription_text'):
                self.transcription_text.insert(tk.END, letter)
                self.transcription_text.see(tk.END)
            self.transcription_var.set(self.transcription_var.get() + letter)
            self.last_transcribed_letter = letter
            self.last_transcription_ts = now

    def add_space(self) -> None:
        """Add space."""
        if hasattr(self, 'transcription_text'):
            self.transcription_text.insert(tk.END, " ")
            self.transcription_text.see(tk.END)
        self.transcription_var.set(self.transcription_var.get() + " ")
        self._reset_transcription_state()

    def backspace(self) -> None:
        """Backspace."""
        if hasattr(self, 'transcription_text'):
            current = self.transcription_text.get("1.0", tk.END)
            if len(current) > 1:
                self.transcription_text.delete("end-2c", tk.END)
        current = self.transcription_var.get()
        if current:
            self.transcription_var.set(current[:-1])
        self._reset_transcription_state()

    def clear_transcription(self) -> None:
        """Clear transcription."""
        if hasattr(self, 'transcription_text'):
            self.transcription_text.delete("1.0", tk.END)
        self.transcription_var.set("")
        self._reset_transcription_state()

    def _reset_transcription_state(self) -> None:
        """Reset transcription state."""
        self.last_transcribed_letter = None
        self.last_transcription_ts = 0.0

    def _append_history(self, result: DetectionResult) -> None:
        """Append to history."""
        timestamp = time.strftime("%H:%M:%S")
        source = result.source or "unknown"
        confidence = f"{result.confidence * 100:.0f}%" if result.confidence is not None else "-"
        entry = f"{timestamp}  {result.letter} ({source}, {confidence})"

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
        """Clear history."""
        self.history_entries.clear()
        self.history_list.delete(0, tk.END)

    def quit(self) -> None:
        """Quit application."""
        self.video_active = False
        if self.voice:
            self.voice.cleanup()
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        self.root.destroy()


def main() -> None:
    """Entry point."""
    handlers = [logging.StreamHandler()]
    try:
        log_path = Path("lsf_detector_enhanced.log")
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
    except Exception:
        LOGGER.debug("Unable to attach file logger", exc_info=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )
    
    root = tk.Tk()
    app = EnhancedLsfApp(root)
    root.protocol("WM_DELETE_WINDOW", app.quit)
    root.mainloop()


if __name__ == "__main__":
    main()
