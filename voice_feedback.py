"""Voice feedback module using pyttsx3 for text-to-speech synthesis.

Provides audio feedback for detected letters, words, and phrases.
"""

from __future__ import annotations

from typing import Optional
from enum import Enum
import threading
import queue
from dataclasses import dataclass


class FeedbackMode(Enum):
    """Voice feedback modes."""
    OFF = "off"  # No voice feedback
    LETTERS = "letters"  # Speak individual letters
    WORDS = "words"  # Speak completed words
    PHRASES = "phrases"  # Speak completed phrases
    ALL = "all"  # Speak everything


@dataclass
class VoiceSettings:
    """Configuration for voice synthesis."""
    enabled: bool = True
    mode: FeedbackMode = FeedbackMode.WORDS
    rate: int = 150  # Words per minute
    volume: float = 0.9  # 0.0 to 1.0
    voice_id: Optional[str] = None  # Specific voice ID (None = default)
    language: str = "fr-FR"  # Voice language code


class VoiceFeedback:
    """Manages text-to-speech feedback for sign language detection."""
    
    def __init__(self, settings: Optional[VoiceSettings] = None):
        """
        Initialize voice feedback system.
        
        Args:
            settings: Voice configuration settings
        """
        self.settings = settings or VoiceSettings()
        self.engine = None
        self.speech_queue = queue.Queue()
        self.is_running = False
        self.speech_thread = None
        
        # Initialize TTS engine
        self._initialize_engine()
        
        # Start speech worker thread
        if self.settings.enabled:
            self.start()
    
    def _initialize_engine(self):
        """Initialize pyttsx3 TTS engine with settings."""
        try:
            import pyttsx3
            self.engine = pyttsx3.init()
            
            # Apply settings
            self.engine.setProperty('rate', self.settings.rate)
            self.engine.setProperty('volume', self.settings.volume)
            
            # Set voice if specified
            if self.settings.voice_id:
                self.engine.setProperty('voice', self.settings.voice_id)
            else:
                # Try to find voice matching language
                voices = self.engine.getProperty('voices')
                for voice in voices:
                    # Check if voice language matches
                    if hasattr(voice, 'languages') and self.settings.language in voice.languages:
                        self.engine.setProperty('voice', voice.id)
                        break
                    # Fallback: check voice id/name
                    elif self.settings.language[:2].lower() in voice.id.lower():
                        self.engine.setProperty('voice', voice.id)
                        break
            
            print(f"Voice feedback initialized: {self.settings.language}")
            
        except ImportError:
            print("Warning: pyttsx3 not installed. Voice feedback disabled.")
            print("Install with: pip install pyttsx3")
            self.engine = None
            self.settings.enabled = False
        except Exception as e:
            print(f"Error initializing voice feedback: {e}")
            self.engine = None
            self.settings.enabled = False
    
    def start(self):
        """Start speech worker thread."""
        if not self.settings.enabled or not self.engine:
            return
        
        if self.is_running:
            return
        
        self.is_running = True
        self.speech_thread = threading.Thread(target=self._speech_worker, daemon=True)
        self.speech_thread.start()
        print("Voice feedback started")
    
    def stop(self):
        """Stop speech worker thread."""
        if not self.is_running:
            return
        
        self.is_running = False
        self.speech_queue.put(None)  # Signal thread to exit
        
        if self.speech_thread:
            self.speech_thread.join(timeout=2.0)
        
        print("Voice feedback stopped")
    
    def _speech_worker(self):
        """Worker thread that processes speech queue."""
        while self.is_running:
            try:
                text = self.speech_queue.get(timeout=0.5)
                
                if text is None:  # Exit signal
                    break
                
                if self.engine and text:
                    self.engine.say(text)
                    self.engine.runAndWait()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in speech worker: {e}")
    
    def speak_letter(self, letter: str):
        """
        Speak a detected letter.
        
        Args:
            letter: Single letter character
        """
        if not self.settings.enabled or self.settings.mode == FeedbackMode.OFF:
            return
        
        if self.settings.mode in (FeedbackMode.LETTERS, FeedbackMode.ALL):
            if letter and len(letter) == 1 and letter.isalpha():
                self.speech_queue.put(letter.upper())
    
    def speak_word(self, word: str):
        """
        Speak a detected word.
        
        Args:
            word: Complete word
        """
        if not self.settings.enabled or self.settings.mode == FeedbackMode.OFF:
            return
        
        if self.settings.mode in (FeedbackMode.WORDS, FeedbackMode.ALL):
            if word:
                self.speech_queue.put(word)
    
    def speak_phrase(self, phrase: str):
        """
        Speak a detected phrase.
        
        Args:
            phrase: Complete phrase or sentence
        """
        if not self.settings.enabled or self.settings.mode == FeedbackMode.OFF:
            return
        
        if self.settings.mode in (FeedbackMode.PHRASES, FeedbackMode.ALL):
            if phrase:
                self.speech_queue.put(phrase)
    
    def speak_custom(self, text: str):
        """
        Speak arbitrary text.
        
        Args:
            text: Any text to speak
        """
        if not self.settings.enabled or not text:
            return
        
        self.speech_queue.put(text)
    
    def set_mode(self, mode: FeedbackMode):
        """Change feedback mode."""
        self.settings.mode = mode
        print(f"Voice feedback mode: {mode.value}")
    
    def set_enabled(self, enabled: bool):
        """Enable or disable voice feedback."""
        was_enabled = self.settings.enabled
        self.settings.enabled = enabled
        
        if enabled and not was_enabled:
            self.start()
        elif not enabled and was_enabled:
            self.stop()
    
    def set_rate(self, rate: int):
        """
        Set speech rate.
        
        Args:
            rate: Words per minute (typical range: 100-200)
        """
        self.settings.rate = max(50, min(300, rate))  # Clamp to reasonable range
        if self.engine:
            self.engine.setProperty('rate', self.settings.rate)
    
    def set_volume(self, volume: float):
        """
        Set speech volume.
        
        Args:
            volume: Volume level (0.0 to 1.0)
        """
        self.settings.volume = max(0.0, min(1.0, volume))
        if self.engine:
            self.engine.setProperty('volume', self.settings.volume)
    
    def get_available_voices(self) -> list[dict]:
        """Get list of available TTS voices."""
        if not self.engine:
            return []
        
        voices = self.engine.getProperty('voices')
        return [
            {
                'id': voice.id,
                'name': voice.name,
                'languages': getattr(voice, 'languages', []),
                'gender': getattr(voice, 'gender', 'unknown')
            }
            for voice in voices
        ]
    
    def set_voice(self, voice_id: str):
        """Set specific voice by ID."""
        self.settings.voice_id = voice_id
        if self.engine:
            try:
                self.engine.setProperty('voice', voice_id)
                print(f"Voice changed to: {voice_id}")
            except Exception as e:
                print(f"Error setting voice: {e}")
    
    def test_voice(self):
        """Test current voice with sample text."""
        test_texts = {
            "fr-FR": "Bonjour, ceci est un test de synthèse vocale pour la langue des signes française.",
            "en-US": "Hello, this is a voice synthesis test for American Sign Language.",
            "en-GB": "Hello, this is a voice synthesis test for British Sign Language.",
            "fr-CA": "Bonjour, ceci est un test de synthèse vocale pour la langue des signes québécoise."
        }
        
        text = test_texts.get(self.settings.language, "Voice test.")
        self.speak_custom(text)
    
    def clear_queue(self):
        """Clear pending speech queue."""
        while not self.speech_queue.empty():
            try:
                self.speech_queue.get_nowait()
            except queue.Empty:
                break
    
    def __del__(self):
        """Cleanup on deletion."""
        self.stop()


class VoiceFeedbackUI:
    """Helper class for creating voice feedback UI controls."""
    
    @staticmethod
    def create_mode_options() -> list[tuple[str, FeedbackMode]]:
        """Get list of feedback mode options for UI."""
        return [
            ("Désactivé / Off", FeedbackMode.OFF),
            ("Lettres / Letters", FeedbackMode.LETTERS),
            ("Mots / Words", FeedbackMode.WORDS),
            ("Phrases / Phrases", FeedbackMode.PHRASES),
            ("Tout / All", FeedbackMode.ALL),
        ]
    
    @staticmethod
    def format_voice_info(voice_info: dict) -> str:
        """Format voice information for display."""
        name = voice_info.get('name', 'Unknown')
        langs = ', '.join(voice_info.get('languages', []))
        gender = voice_info.get('gender', 'unknown')
        
        return f"{name} ({langs}) - {gender}"


if __name__ == "__main__":
    # Demo usage
    print("Voice Feedback Demo")
    print("=" * 50)
    
    # Create voice feedback with French settings
    settings = VoiceSettings(
        enabled=True,
        mode=FeedbackMode.ALL,
        rate=150,
        volume=0.9,
        language="fr-FR"
    )
    
    voice = VoiceFeedback(settings)
    
    # List available voices
    print("\nAvailable voices:")
    for v in voice.get_available_voices()[:5]:  # Show first 5
        print(f"  - {VoiceFeedbackUI.format_voice_info(v)}")
    
    # Test voice
    print("\nTesting voice feedback...")
    voice.test_voice()
    
    import time
    time.sleep(2)
    
    # Simulate detection sequence
    print("\nSimulating letter detection:")
    for letter in "BONJOUR":
        print(f"  Detected: {letter}")
        voice.speak_letter(letter)
        time.sleep(0.5)
    
    time.sleep(2)
    
    # Simulate word detection
    print("\nSimulating word detection:")
    voice.speak_word("BONJOUR")
    
    time.sleep(3)
    
    # Change mode
    print("\nSwitching to words-only mode...")
    voice.set_mode(FeedbackMode.WORDS)
    
    print("\nSpeaking a phrase:")
    voice.speak_phrase("La langue des signes est fascinante")
    
    time.sleep(5)
    
    # Cleanup
    voice.stop()
    print("\nDemo complete")
