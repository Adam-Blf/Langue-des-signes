"""Word and phrase detection module for LSF.

This module segments detected letters into words using temporal analysis
and validates against a French dictionary.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional, List, Set
import time
import json
from pathlib import Path


@dataclass
class LetterFrame:
    """Represents a detected letter at a specific time."""
    letter: str
    timestamp: float
    confidence: float = 1.0


class WordDetector:
    """Detects words from a stream of letters using temporal segmentation."""
    
    def __init__(
        self,
        pause_threshold: float = 1.5,
        min_word_length: int = 2,
        dictionary_path: Optional[str] = None
    ):
        """
        Initialize word detector.
        
        Args:
            pause_threshold: Seconds of no detection to trigger word boundary
            min_word_length: Minimum letters required to form a word
            dictionary_path: Path to French word dictionary JSON file
        """
        self.pause_threshold = pause_threshold
        self.min_word_length = min_word_length
        self.letter_buffer: deque[LetterFrame] = deque(maxlen=100)
        self.current_word: List[str] = []
        self.last_detection_time: Optional[float] = None
        self.detected_words: List[str] = []
        self.dictionary: Set[str] = self._load_dictionary(dictionary_path)
        
    def _load_dictionary(self, dictionary_path: Optional[str]) -> Set[str]:
        """Load French word dictionary from JSON file."""
        if dictionary_path is None:
            # Default common French words for LSF
            return {
                "bonjour", "merci", "oui", "non", "salut", "bonsoir",
                "au revoir", "pardon", "excusez-moi", "aide", "help",
                "chat", "chien", "maison", "famille", "ami", "amie",
                "maman", "papa", "frère", "sœur", "eau", "pain",
                "je", "tu", "il", "elle", "nous", "vous", "ils", "elles",
                "aimer", "vouloir", "pouvoir", "aller", "venir", "faire",
                "voir", "dire", "donner", "prendre", "manger", "boire",
                "comprendre", "savoir", "parler", "écouter", "regarder"
            }
        
        try:
            with open(dictionary_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return set(word.lower() for word in data.get('words', []))
        except (FileNotFoundError, json.JSONDecodeError):
            print(f"Warning: Could not load dictionary from {dictionary_path}")
            return set()
    
    def add_letter(self, letter: str, confidence: float = 1.0) -> Optional[str]:
        """
        Add a detected letter to the buffer.
        
        Returns completed word if word boundary detected, None otherwise.
        """
        current_time = time.time()
        
        # Check for word boundary (pause in detections)
        if self.last_detection_time is not None:
            time_since_last = current_time - self.last_detection_time
            
            if time_since_last > self.pause_threshold:
                # Word boundary detected
                completed_word = self._finalize_word()
                if completed_word:
                    self.detected_words.append(completed_word)
                    return completed_word
        
        # Add letter to current word
        if letter and letter.isalpha():
            self.current_word.append(letter.upper())
            self.letter_buffer.append(LetterFrame(letter.upper(), current_time, confidence))
            self.last_detection_time = current_time
        
        return None
    
    def _finalize_word(self) -> Optional[str]:
        """Convert current letter buffer to word string."""
        if len(self.current_word) < self.min_word_length:
            self.current_word.clear()
            return None
        
        word = ''.join(self.current_word)
        self.current_word.clear()
        return word
    
    def force_word_boundary(self) -> Optional[str]:
        """Manually trigger word boundary (e.g., space key pressed)."""
        return self._finalize_word()
    
    def is_valid_word(self, word: str) -> bool:
        """Check if word exists in dictionary."""
        if not self.dictionary:
            return True  # No dictionary loaded, accept all words
        return word.lower() in self.dictionary
    
    def get_word_suggestions(self, partial_word: str, max_suggestions: int = 5) -> List[str]:
        """Get dictionary suggestions for partial word."""
        if not self.dictionary or not partial_word:
            return []
        
        partial_lower = partial_word.lower()
        suggestions = [
            word for word in self.dictionary
            if word.startswith(partial_lower)
        ]
        return sorted(suggestions)[:max_suggestions]
    
    def get_recent_words(self, count: int = 5) -> List[str]:
        """Get the most recently detected words."""
        return self.detected_words[-count:]
    
    def clear_buffer(self):
        """Clear all buffers and reset state."""
        self.letter_buffer.clear()
        self.current_word.clear()
        self.last_detection_time = None
    
    def get_current_partial_word(self) -> str:
        """Get the current word being formed."""
        return ''.join(self.current_word)


class PhraseBuilder:
    """Builds phrases from detected words with grammar awareness."""
    
    def __init__(self, max_phrase_length: int = 10):
        """
        Initialize phrase builder.
        
        Args:
            max_phrase_length: Maximum number of words in a phrase
        """
        self.max_phrase_length = max_phrase_length
        self.current_phrase: List[str] = []
        self.completed_phrases: List[str] = []
        
        # Common LSF phrase patterns (subject-object-verb order)
        self.phrase_patterns = {
            "greeting": ["bonjour", "salut", "bonsoir"],
            "question": ["quoi", "qui", "où", "quand", "comment", "pourquoi"],
            "politeness": ["merci", "s'il vous plaît", "pardon", "excusez-moi"]
        }
    
    def add_word(self, word: str) -> Optional[str]:
        """
        Add word to current phrase.
        
        Returns completed phrase if sentence boundary detected.
        """
        if not word:
            return None
        
        self.current_phrase.append(word)
        
        # Check for phrase completion
        if self._is_phrase_complete():
            phrase = self._finalize_phrase()
            if phrase:
                self.completed_phrases.append(phrase)
                return phrase
        
        # Limit phrase length
        if len(self.current_phrase) >= self.max_phrase_length:
            phrase = self._finalize_phrase()
            if phrase:
                self.completed_phrases.append(phrase)
                return phrase
        
        return None
    
    def _is_phrase_complete(self) -> bool:
        """Check if current phrase forms a complete thought."""
        if not self.current_phrase:
            return False
        
        # Check for punctuation indicators or common ending patterns
        last_word = self.current_phrase[-1].lower()
        
        # Greeting patterns are complete
        if last_word in self.phrase_patterns["greeting"]:
            return True
        
        # Politeness expressions are complete
        if last_word in self.phrase_patterns["politeness"]:
            return True
        
        # Simple sentence pattern: subject + verb (minimum phrase)
        if len(self.current_phrase) >= 2:
            # Basic French verb endings suggest completion
            if last_word.endswith(('er', 'ir', 're', 'ez', 'ons', 'ent')):
                return True
        
        return False
    
    def _finalize_phrase(self) -> Optional[str]:
        """Convert current word buffer to phrase string."""
        if not self.current_phrase:
            return None
        
        phrase = ' '.join(self.current_phrase)
        self.current_phrase.clear()
        return phrase
    
    def force_phrase_boundary(self) -> Optional[str]:
        """Manually trigger phrase boundary (e.g., period key pressed)."""
        return self._finalize_phrase()
    
    def get_current_phrase(self) -> str:
        """Get the current phrase being formed."""
        return ' '.join(self.current_phrase)
    
    def get_recent_phrases(self, count: int = 3) -> List[str]:
        """Get the most recently completed phrases."""
        return self.completed_phrases[-count:]
    
    def clear_buffer(self):
        """Clear phrase buffer."""
        self.current_phrase.clear()


def create_default_dictionary(output_path: str = "lsf_dictionary.json"):
    """Create a default French LSF dictionary file."""
    common_words = [
        # Greetings
        "bonjour", "salut", "bonsoir", "bonne nuit", "au revoir", "à bientôt",
        
        # Politeness
        "merci", "s'il vous plaît", "de rien", "pardon", "excusez-moi",
        "désolé", "bienvenue",
        
        # Basic verbs
        "être", "avoir", "faire", "aller", "venir", "voir", "dire",
        "donner", "prendre", "mettre", "savoir", "pouvoir", "vouloir",
        "devoir", "comprendre", "parler", "écouter", "regarder", "sentir",
        "toucher", "manger", "boire", "dormir", "travailler", "étudier",
        
        # Pronouns
        "je", "tu", "il", "elle", "nous", "vous", "ils", "elles",
        "moi", "toi", "lui", "soi", "on",
        
        # Family
        "famille", "maman", "papa", "mère", "père", "frère", "sœur",
        "enfant", "fils", "fille", "grand-père", "grand-mère", "oncle",
        "tante", "cousin", "cousine",
        
        # Common nouns
        "maison", "école", "travail", "voiture", "vélo", "train", "bus",
        "chat", "chien", "oiseau", "arbre", "fleur", "eau", "pain",
        "livre", "téléphone", "ordinateur", "table", "chaise", "porte",
        "fenêtre", "lit", "jour", "nuit", "temps", "heure",
        
        # Adjectives
        "bon", "mauvais", "grand", "petit", "beau", "joli", "nouveau",
        "vieux", "jeune", "chaud", "froid", "facile", "difficile",
        "content", "triste", "heureux", "fatigué",
        
        # Questions
        "quoi", "qui", "où", "quand", "comment", "pourquoi", "combien",
        
        # Numbers (text form)
        "un", "deux", "trois", "quatre", "cinq", "six", "sept", "huit",
        "neuf", "dix", "cent", "mille",
        
        # Emotions
        "aimer", "adorer", "détester", "préférer", "peur", "joie",
        "colère", "surprise", "amour", "ami", "amie",
        
        # Yes/No
        "oui", "non", "peut-être", "jamais", "toujours", "souvent"
    ]
    
    dictionary_data = {
        "language": "fr",
        "type": "LSF",
        "version": "1.0",
        "words": sorted(common_words)
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dictionary_data, f, ensure_ascii=False, indent=2)
    
    print(f"Created dictionary with {len(common_words)} words at {output_path}")


if __name__ == "__main__":
    # Demo usage
    print("LSF Word Detector Demo")
    print("=" * 50)
    
    # Create default dictionary
    create_default_dictionary()
    
    # Initialize detectors
    word_detector = WordDetector(dictionary_path="lsf_dictionary.json")
    phrase_builder = PhraseBuilder()
    
    # Simulate letter detection stream
    test_sequence = [
        ("B", 0.0), ("O", 0.5), ("N", 1.0), ("J", 1.5), ("O", 2.0),
        ("U", 2.5), ("R", 3.0),  # "BONJOUR"
        (None, 5.0),  # Pause (word boundary)
        ("M", 6.0), ("E", 6.5), ("R", 7.0), ("C", 7.5), ("I", 8.0),  # "MERCI"
    ]
    
    print("\nProcessing letter sequence:")
    for letter, timestamp in test_sequence:
        if letter:
            print(f"  {timestamp}s: Detected '{letter}'")
            word = word_detector.add_letter(letter)
            if word:
                print(f"  → Word completed: '{word}'")
                is_valid = word_detector.is_valid_word(word)
                print(f"    Valid: {is_valid}")
                
                phrase = phrase_builder.add_word(word)
                if phrase:
                    print(f"  → Phrase completed: '{phrase}'")
        else:
            time.sleep(0.1)  # Simulate pause
    
    # Force final word
    final_word = word_detector.force_word_boundary()
    if final_word:
        print(f"\nFinal word: '{final_word}'")
    
    print(f"\nDetected words: {word_detector.get_recent_words()}")
    print(f"Completed phrases: {phrase_builder.get_recent_phrases()}")
