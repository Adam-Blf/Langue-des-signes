"""Multi-language configuration for sign language detection.

Supports LSF (French), ASL (American), LSQ (Quebec), and other sign languages.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Callable, Any
from pathlib import Path
import json


class SignLanguage(Enum):
    """Supported sign languages."""
    LSF = "lsf"  # Langue des Signes Française (French Sign Language)
    ASL = "asl"  # American Sign Language
    LSQ = "lsq"  # Langue des Signes Québécoise (Quebec Sign Language)
    BSL = "bsl"  # British Sign Language
    AUSLAN = "auslan"  # Australian Sign Language
    ISL = "isl"  # Irish Sign Language
    DGS = "dgs"  # Deutsche Gebärdensprache (German Sign Language)


@dataclass
class LanguageConfig:
    """Configuration for a specific sign language."""
    
    code: str
    display_name: str
    full_name: str
    country_code: str
    model_path: Optional[str] = None
    dictionary_path: Optional[str] = None
    heuristic_module: Optional[str] = None
    
    # Alphabet characteristics
    has_one_handed_alphabet: bool = True
    has_two_handed_alphabet: bool = False
    alphabet_letters: int = 26
    
    # Voice synthesis
    voice_language_code: str = "en-US"
    
    # UI translations
    ui_translations: Dict[str, str] = None
    
    def __post_init__(self):
        if self.ui_translations is None:
            self.ui_translations = {}


# Language configurations
LANGUAGE_CONFIGS: Dict[SignLanguage, LanguageConfig] = {
    SignLanguage.LSF: LanguageConfig(
        code="lsf",
        display_name="LSF",
        full_name="Langue des Signes Française",
        country_code="FR",
        model_path="models/lsf_model.pkl",
        dictionary_path="dictionaries/lsf_dictionary.json",
        heuristic_module="letters_conditions_extended",
        has_one_handed_alphabet=True,
        has_two_handed_alphabet=False,
        alphabet_letters=26,
        voice_language_code="fr-FR",
        ui_translations={
            "title": "Détection Langue des Signes Française",
            "start": "Démarrer",
            "stop": "Arrêter",
            "settings": "Paramètres",
            "language": "Langue",
            "confidence": "Confiance",
            "detected_letter": "Lettre détectée",
            "detected_word": "Mot détecté",
            "transcription": "Transcription"
        }
    ),
    
    SignLanguage.ASL: LanguageConfig(
        code="asl",
        display_name="ASL",
        full_name="American Sign Language",
        country_code="US",
        model_path="models/asl_model.pkl",
        dictionary_path="dictionaries/asl_dictionary.json",
        heuristic_module="letters_conditions_asl",
        has_one_handed_alphabet=True,
        has_two_handed_alphabet=False,
        alphabet_letters=26,
        voice_language_code="en-US",
        ui_translations={
            "title": "American Sign Language Detection",
            "start": "Start",
            "stop": "Stop",
            "settings": "Settings",
            "language": "Language",
            "confidence": "Confidence",
            "detected_letter": "Detected Letter",
            "detected_word": "Detected Word",
            "transcription": "Transcription"
        }
    ),
    
    SignLanguage.LSQ: LanguageConfig(
        code="lsq",
        display_name="LSQ",
        full_name="Langue des Signes Québécoise",
        country_code="CA",
        model_path="models/lsq_model.pkl",
        dictionary_path="dictionaries/lsq_dictionary.json",
        heuristic_module="letters_conditions_lsq",
        has_one_handed_alphabet=True,
        has_two_handed_alphabet=False,
        alphabet_letters=26,
        voice_language_code="fr-CA",
        ui_translations={
            "title": "Détection Langue des Signes Québécoise",
            "start": "Démarrer",
            "stop": "Arrêter",
            "settings": "Paramètres",
            "language": "Langue",
            "confidence": "Confiance",
            "detected_letter": "Lettre détectée",
            "detected_word": "Mot détecté",
            "transcription": "Transcription"
        }
    ),
    
    SignLanguage.BSL: LanguageConfig(
        code="bsl",
        display_name="BSL",
        full_name="British Sign Language",
        country_code="GB",
        model_path="models/bsl_model.pkl",
        dictionary_path="dictionaries/bsl_dictionary.json",
        heuristic_module="letters_conditions_bsl",
        has_one_handed_alphabet=False,
        has_two_handed_alphabet=True,
        alphabet_letters=26,
        voice_language_code="en-GB",
        ui_translations={
            "title": "British Sign Language Detection",
            "start": "Start",
            "stop": "Stop",
            "settings": "Settings",
            "language": "Language",
            "confidence": "Confidence",
            "detected_letter": "Detected Letter",
            "detected_word": "Detected Word",
            "transcription": "Transcription"
        }
    ),
}


class LanguageManager:
    """Manages sign language selection and configuration loading."""
    
    def __init__(self, default_language: SignLanguage = SignLanguage.LSF):
        """
        Initialize language manager.
        
        Args:
            default_language: The default sign language to use
        """
        self.current_language = default_language
        self.config = LANGUAGE_CONFIGS[default_language]
        self.model_cache: Dict[str, Any] = {}
        self.dictionary_cache: Dict[str, set] = {}
    
    def set_language(self, language: SignLanguage):
        """Switch to a different sign language."""
        if language not in LANGUAGE_CONFIGS:
            raise ValueError(f"Unsupported language: {language}")
        
        self.current_language = language
        self.config = LANGUAGE_CONFIGS[language]
        print(f"Switched to {self.config.full_name}")
    
    def get_config(self) -> LanguageConfig:
        """Get configuration for current language."""
        return self.config
    
    def get_ui_text(self, key: str) -> str:
        """Get translated UI text for current language."""
        return self.config.ui_translations.get(key, key)
    
    def load_model(self) -> Optional[Any]:
        """Load ML model for current language (cached)."""
        if not self.config.model_path:
            return None
        
        if self.config.code in self.model_cache:
            return self.model_cache[self.config.code]
        
        model_path = Path(self.config.model_path)
        if not model_path.exists():
            print(f"Warning: Model not found at {model_path}")
            return None
        
        try:
            import joblib
            model = joblib.load(model_path)
            self.model_cache[self.config.code] = model
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    
    def load_dictionary(self) -> set:
        """Load word dictionary for current language (cached)."""
        if not self.config.dictionary_path:
            return set()
        
        if self.config.code in self.dictionary_cache:
            return self.dictionary_cache[self.config.code]
        
        dict_path = Path(self.config.dictionary_path)
        if not dict_path.exists():
            print(f"Warning: Dictionary not found at {dict_path}")
            return set()
        
        try:
            with open(dict_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                words = set(word.lower() for word in data.get('words', []))
                self.dictionary_cache[self.config.code] = words
                return words
        except Exception as e:
            print(f"Error loading dictionary: {e}")
            return set()
    
    def get_heuristic_detector(self) -> Optional[Callable]:
        """Get heuristic letter detection function for current language."""
        if not self.config.heuristic_module:
            return None
        
        try:
            module = __import__(self.config.heuristic_module)
            
            # Try extended detector first, fallback to basic
            if hasattr(module, 'detect_letter_extended'):
                return module.detect_letter_extended
            elif hasattr(module, 'detect_letter'):
                return module.detect_letter
            else:
                print(f"Warning: No detector found in {self.config.heuristic_module}")
                return None
        except ImportError as e:
            print(f"Warning: Could not import {self.config.heuristic_module}: {e}")
            return None
    
    def list_available_languages(self) -> list[tuple[SignLanguage, str]]:
        """Get list of all supported languages."""
        return [
            (lang, config.full_name)
            for lang, config in LANGUAGE_CONFIGS.items()
        ]
    
    def export_config(self, output_path: str):
        """Export current configuration to JSON file."""
        config_dict = {
            "language": self.config.code,
            "display_name": self.config.display_name,
            "full_name": self.config.full_name,
            "country_code": self.config.country_code,
            "model_path": self.config.model_path,
            "dictionary_path": self.config.dictionary_path,
            "voice_language_code": self.config.voice_language_code,
            "alphabet_characteristics": {
                "one_handed": self.config.has_one_handed_alphabet,
                "two_handed": self.config.has_two_handed_alphabet,
                "letters": self.config.alphabet_letters
            },
            "ui_translations": self.config.ui_translations
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, ensure_ascii=False, indent=2)
        
        print(f"Exported configuration to {output_path}")


def create_language_selector_ui() -> str:
    """Generate HTML/text for language selection UI."""
    manager = LanguageManager()
    languages = manager.list_available_languages()
    
    ui_text = "Available Sign Languages:\n"
    ui_text += "=" * 50 + "\n"
    
    for i, (lang, full_name) in enumerate(languages, 1):
        config = LANGUAGE_CONFIGS[lang]
        ui_text += f"{i}. {config.display_name} - {full_name} ({config.country_code})\n"
        ui_text += f"   Alphabet: {config.alphabet_letters} letters, "
        ui_text += f"{'One-handed' if config.has_one_handed_alphabet else 'Two-handed'}\n"
    
    return ui_text


if __name__ == "__main__":
    # Demo usage
    print("Multi-Language Sign Detection Demo")
    print("=" * 50)
    
    # Create language manager with LSF
    manager = LanguageManager(SignLanguage.LSF)
    print(f"\nCurrent language: {manager.config.full_name}")
    print(f"UI Title: {manager.get_ui_text('title')}")
    print(f"Start button: {manager.get_ui_text('start')}")
    
    # Switch to ASL
    print("\n" + "=" * 50)
    manager.set_language(SignLanguage.ASL)
    print(f"\nCurrent language: {manager.config.full_name}")
    print(f"UI Title: {manager.get_ui_text('title')}")
    print(f"Start button: {manager.get_ui_text('start')}")
    
    # List all languages
    print("\n" + "=" * 50)
    print(create_language_selector_ui())
    
    # Export configuration
    manager.export_config("language_config_asl.json")
