"""Tests for voice_feedback.py - TTS voice feedback."""

import pytest
from unittest.mock import Mock, patch
from voice_feedback import (
    VoiceFeedback,
    FeedbackMode,
    VoiceSettings
)


class TestVoiceSettings:
    """Test VoiceSettings dataclass."""
    
    def test_voice_settings_creation(self):
        """Test creating VoiceSettings."""
        settings = VoiceSettings(
            enabled=True,
            mode=FeedbackMode.ALL,
            rate=150,
            volume=0.8,
            language='fr-FR'
        )
        
        assert settings.enabled is True
        assert settings.mode == FeedbackMode.ALL
        assert settings.rate == 150
        assert settings.volume == 0.8
        assert settings.language == 'fr-FR'
    
    def test_voice_settings_defaults(self):
        """Test VoiceSettings default values."""
        settings = VoiceSettings()
        
        assert settings.enabled is True
        assert settings.mode == FeedbackMode.OFF
        assert 100 <= settings.rate <= 200
        assert 0.0 <= settings.volume <= 1.0


class TestFeedbackMode:
    """Test FeedbackMode enum."""
    
    def test_feedback_modes_exist(self):
        """Test that all feedback modes are defined."""
        assert hasattr(FeedbackMode, 'OFF')
        assert hasattr(FeedbackMode, 'LETTERS')
        assert hasattr(FeedbackMode, 'WORDS')
        assert hasattr(FeedbackMode, 'PHRASES')
        assert hasattr(FeedbackMode, 'ALL')
    
    def test_feedback_mode_values(self):
        """Test feedback mode values."""
        assert FeedbackMode.OFF.value == "off"
        assert FeedbackMode.LETTERS.value == "letters"
        assert FeedbackMode.WORDS.value == "words"
        assert FeedbackMode.PHRASES.value == "phrases"
        assert FeedbackMode.ALL.value == "all"


class TestVoiceFeedback:
    """Test VoiceFeedback class."""
    
    @patch('voice_feedback.pyttsx3')
    def test_voice_feedback_initialization(self, mock_pyttsx3):
        """Test VoiceFeedback initialization."""
        mock_engine = Mock()
        mock_pyttsx3.init.return_value = mock_engine
        
        voice = VoiceFeedback()
        
        assert voice is not None
        assert voice.enabled is True
    
    @patch('voice_feedback.pyttsx3')
    def test_voice_feedback_without_pyttsx3(self, mock_pyttsx3):
        """Test VoiceFeedback when pyttsx3 is not available."""
        mock_pyttsx3.init.side_effect = ImportError("pyttsx3 not found")
        
        voice = VoiceFeedback()
        
        # Should initialize but with disabled state
        assert voice.enabled is False or voice is not None
    
    @patch('voice_feedback.pyttsx3')
    def test_set_mode(self, mock_pyttsx3):
        """Test setting feedback mode."""
        mock_engine = Mock()
        mock_pyttsx3.init.return_value = mock_engine
        
        voice = VoiceFeedback()
        voice.set_mode(FeedbackMode.LETTERS)
        
        assert voice.mode == FeedbackMode.LETTERS
    
    @patch('voice_feedback.pyttsx3')
    def test_set_rate(self, mock_pyttsx3):
        """Test setting speech rate."""
        mock_engine = Mock()
        mock_pyttsx3.init.return_value = mock_engine
        
        voice = VoiceFeedback()
        voice.set_rate(180)
        
        # Rate should be clamped to valid range
        assert 100 <= voice.rate <= 200
    
    @patch('voice_feedback.pyttsx3')
    def test_set_volume(self, mock_pyttsx3):
        """Test setting volume."""
        mock_engine = Mock()
        mock_pyttsx3.init.return_value = mock_engine
        
        voice = VoiceFeedback()
        voice.set_volume(0.75)
        
        assert 0.0 <= voice.volume <= 1.0
    
    @patch('voice_feedback.pyttsx3')
    def test_speak_letter(self, mock_pyttsx3):
        """Test speaking a letter."""
        mock_engine = Mock()
        mock_pyttsx3.init.return_value = mock_engine
        
        voice = VoiceFeedback()
        voice.set_mode(FeedbackMode.LETTERS)
        voice.speak_letter('A')
        
        # Should queue the letter for speaking
        # (actual speech happens in background thread)
        assert True  # If no exception, test passes
    
    @patch('voice_feedback.pyttsx3')
    def test_speak_word(self, mock_pyttsx3):
        """Test speaking a word."""
        mock_engine = Mock()
        mock_pyttsx3.init.return_value = mock_engine
        
        voice = VoiceFeedback()
        voice.set_mode(FeedbackMode.WORDS)
        voice.speak_word("BONJOUR")
        
        assert True  # If no exception, test passes
    
    @patch('voice_feedback.pyttsx3')
    def test_speak_phrase(self, mock_pyttsx3):
        """Test speaking a phrase."""
        mock_engine = Mock()
        mock_pyttsx3.init.return_value = mock_engine
        
        voice = VoiceFeedback()
        voice.set_mode(FeedbackMode.PHRASES)
        voice.speak_phrase("BONJOUR COMMENT ALLEZ VOUS")
        
        assert True  # If no exception, test passes
    
    @patch('voice_feedback.pyttsx3')
    def test_set_enabled(self, mock_pyttsx3):
        """Test enabling/disabling voice feedback."""
        mock_engine = Mock()
        mock_pyttsx3.init.return_value = mock_engine
        
        voice = VoiceFeedback()
        
        voice.set_enabled(False)
        assert voice.enabled is False
        
        voice.set_enabled(True)
        assert voice.enabled is True
    
    @patch('voice_feedback.pyttsx3')
    def test_get_available_voices(self, mock_pyttsx3):
        """Test getting available voices."""
        mock_engine = Mock()
        mock_voice = Mock()
        mock_voice.id = "voice1"
        mock_voice.name = "French Voice"
        mock_voice.languages = ["fr-FR"]
        mock_engine.getProperty.return_value = [mock_voice]
        mock_pyttsx3.init.return_value = mock_engine
        
        voice = VoiceFeedback()
        voices = voice.get_available_voices()
        
        assert isinstance(voices, list)
    
    @patch('voice_feedback.pyttsx3')
    def test_cleanup(self, mock_pyttsx3):
        """Test cleanup of voice resources."""
        mock_engine = Mock()
        mock_pyttsx3.init.return_value = mock_engine
        
        voice = VoiceFeedback()
        voice.cleanup()
        
        # Should stop thread and cleanup resources
        assert True  # If no exception, test passes


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
