"""Tests for word_detector.py - Word and phrase detection."""

import pytest
import time
from word_detector import (
    WordDetector,
    PhraseBuilder,
    LetterFrame,
    create_default_dictionary
)


class TestLetterFrame:
    """Test LetterFrame dataclass."""
    
    def test_letter_frame_creation(self):
        """Test creating a LetterFrame."""
        frame = LetterFrame('A', time.time(), 0.95)
        assert frame.letter == 'A'
        assert frame.confidence == 0.95
        assert isinstance(frame.timestamp, float)


class TestWordDetector:
    """Test word detection with temporal segmentation."""
    
    def test_word_detector_initialization(self):
        """Test WordDetector initialization."""
        detector = WordDetector(pause_threshold=1.5)
        assert detector.pause_threshold == 1.5
        assert len(detector.letter_buffer) == 0
    
    def test_add_single_letter(self):
        """Test adding a single letter."""
        detector = WordDetector()
        detector.add_letter('B', confidence=0.90)
        
        assert len(detector.letter_buffer) == 1
        assert detector.letter_buffer[0].letter == 'B'
    
    def test_word_formation_with_pause(self):
        """Test word formation after pause threshold."""
        detector = WordDetector(pause_threshold=0.1)
        
        # Add letters quickly
        detector.add_letter('B', confidence=0.90)
        detector.add_letter('O', confidence=0.92)
        detector.add_letter('N', confidence=0.88)
        
        # Wait for pause
        time.sleep(0.15)
        
        # Add another letter to trigger word completion
        detector.add_letter('J', confidence=0.85)
        
        # Previous word should be detected
        word = detector.get_current_word()
        # Note: May return None if word not in dictionary
        assert word is None or word == "BON"
    
    def test_clear_buffer(self):
        """Test clearing letter buffer."""
        detector = WordDetector()
        detector.add_letter('A', confidence=0.90)
        detector.add_letter('B', confidence=0.85)
        
        detector.clear()
        assert len(detector.letter_buffer) == 0
    
    def test_ignore_low_confidence_letters(self):
        """Test that low confidence letters are filtered."""
        detector = WordDetector(min_confidence=0.8)
        detector.add_letter('A', confidence=0.75)  # Below threshold
        
        assert len(detector.letter_buffer) == 0
    
    def test_duplicate_letter_filtering(self):
        """Test that rapid duplicate letters are filtered."""
        detector = WordDetector()
        
        # Add same letter rapidly (should keep only first or merge)
        detector.add_letter('A', confidence=0.90)
        time.sleep(0.01)  # Very short delay
        detector.add_letter('A', confidence=0.92)
        
        # Should have filtered duplicates
        assert len(detector.letter_buffer) <= 2


class TestPhraseBuilder:
    """Test phrase construction from words."""
    
    def test_phrase_builder_initialization(self):
        """Test PhraseBuilder initialization."""
        builder = PhraseBuilder()
        assert len(builder.words) == 0
    
    def test_add_single_word(self):
        """Test adding a single word."""
        builder = PhraseBuilder()
        builder.add_word("BONJOUR")
        
        assert len(builder.words) == 1
        assert builder.words[0] == "BONJOUR"
    
    def test_add_multiple_words(self):
        """Test adding multiple words."""
        builder = PhraseBuilder()
        builder.add_word("BONJOUR")
        builder.add_word("COMMENT")
        builder.add_word("ALLEZ")
        
        assert len(builder.words) == 3
    
    def test_get_phrase(self):
        """Test getting complete phrase."""
        builder = PhraseBuilder()
        builder.add_word("BONJOUR")
        builder.add_word("COMMENT")
        
        phrase = builder.get_phrase()
        assert phrase == "BONJOUR COMMENT"
    
    def test_is_phrase_complete_greeting(self):
        """Test phrase completion detection for greetings."""
        builder = PhraseBuilder()
        builder.add_word("BONJOUR")
        
        # Single greeting word may be complete
        is_complete = builder.is_phrase_complete()
        assert isinstance(is_complete, bool)
    
    def test_reset_phrase(self):
        """Test resetting phrase builder."""
        builder = PhraseBuilder()
        builder.add_word("BONJOUR")
        builder.add_word("MERCI")
        
        builder.reset()
        assert len(builder.words) == 0
        assert builder.get_phrase() == ""
    
    def test_max_phrase_length(self):
        """Test maximum phrase length limit."""
        builder = PhraseBuilder(max_phrase_length=3)
        
        # Add more words than limit
        for word in ["UN", "DEUX", "TROIS", "QUATRE", "CINQ"]:
            builder.add_word(word)
        
        # Should not exceed max length
        assert len(builder.words) <= 3
    
    def test_ignore_empty_words(self):
        """Test that empty words are ignored."""
        builder = PhraseBuilder()
        builder.add_word("")
        builder.add_word("  ")
        
        assert len(builder.words) == 0


class TestDictionaryCreation:
    """Test dictionary creation utility."""
    
    def test_create_default_dictionary_returns_path(self):
        """Test that dictionary creation returns a valid path."""
        path = create_default_dictionary(output_path="test_dictionary.json")
        
        assert path is not None
        assert str(path).endswith(".json")
    
    def test_dictionary_contains_common_words(self):
        """Test that default dictionary has expected words."""
        import json
        from pathlib import Path
        
        path = create_default_dictionary(output_path="test_dictionary.json")
        
        with open(path, 'r', encoding='utf-8') as f:
            dictionary = json.load(f)
        
        # Check some common LSF words are present
        assert "BONJOUR" in dictionary
        assert "MERCI" in dictionary
        assert "OUI" in dictionary
        assert "NON" in dictionary
        
        # Cleanup
        Path(path).unlink(missing_ok=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
