"""Interactive learning mode for sign language practice.

Provides guided exercises, progress tracking, and real-time feedback.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict
from datetime import datetime
import json
import random
from pathlib import Path


class ExerciseType(Enum):
    """Types of learning exercises."""
    ALPHABET_RECOGNITION = "alphabet_recognition"  # Practice individual letters
    WORD_SPELLING = "word_spelling"  # Spell complete words
    PHRASE_PRACTICE = "phrase_practice"  # Practice common phrases
    SPEED_CHALLENGE = "speed_challenge"  # Timed exercises
    QUIZ = "quiz"  # Test knowledge


class DifficultyLevel(Enum):
    """Difficulty levels for exercises."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


@dataclass
class Exercise:
    """Represents a single learning exercise."""
    
    id: str
    type: ExerciseType
    difficulty: DifficultyLevel
    title: str
    description: str
    target: str  # Target letter, word, or phrase
    hints: List[str] = field(default_factory=list)
    time_limit: Optional[int] = None  # Seconds (None = unlimited)
    required_repetitions: int = 3  # Times to successfully complete
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "type": self.type.value,
            "difficulty": self.difficulty.value,
            "title": self.title,
            "description": self.description,
            "target": self.target,
            "hints": self.hints,
            "time_limit": self.time_limit,
            "required_repetitions": self.required_repetitions
        }


@dataclass
class ExerciseResult:
    """Results from completing an exercise."""
    
    exercise_id: str
    completed: bool
    attempts: int
    time_taken: float  # Seconds
    accuracy: float  # 0.0 to 1.0
    errors: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "exercise_id": self.exercise_id,
            "completed": self.completed,
            "attempts": self.attempts,
            "time_taken": self.time_taken,
            "accuracy": self.accuracy,
            "errors": self.errors,
            "timestamp": self.timestamp
        }


@dataclass
class UserProgress:
    """Tracks user learning progress."""
    
    user_id: str
    current_level: DifficultyLevel = DifficultyLevel.BEGINNER
    completed_exercises: List[str] = field(default_factory=list)
    exercise_results: List[ExerciseResult] = field(default_factory=list)
    mastered_letters: List[str] = field(default_factory=list)
    mastered_words: List[str] = field(default_factory=list)
    total_practice_time: float = 0.0  # Minutes
    streak_days: int = 0
    last_practice_date: Optional[str] = None
    
    def add_result(self, result: ExerciseResult):
        """Add exercise result and update progress."""
        self.exercise_results.append(result)
        
        if result.completed and result.exercise_id not in self.completed_exercises:
            self.completed_exercises.append(result.exercise_id)
        
        self.total_practice_time += result.time_taken / 60.0  # Convert to minutes
    
    def get_accuracy_trend(self, last_n: int = 10) -> float:
        """Calculate average accuracy from recent exercises."""
        recent = self.exercise_results[-last_n:]
        if not recent:
            return 0.0
        return sum(r.accuracy for r in recent) / len(recent)
    
    def should_level_up(self) -> bool:
        """Determine if user should advance to next difficulty level."""
        # Criteria: 80%+ accuracy on last 10 exercises
        if len(self.exercise_results) < 10:
            return False
        
        recent_accuracy = self.get_accuracy_trend(10)
        return recent_accuracy >= 0.80
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "user_id": self.user_id,
            "current_level": self.current_level.value,
            "completed_exercises": self.completed_exercises,
            "exercise_results": [r.to_dict() for r in self.exercise_results],
            "mastered_letters": self.mastered_letters,
            "mastered_words": self.mastered_words,
            "total_practice_time": self.total_practice_time,
            "streak_days": self.streak_days,
            "last_practice_date": self.last_practice_date
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> UserProgress:
        """Create from dictionary."""
        progress = cls(user_id=data["user_id"])
        progress.current_level = DifficultyLevel(data.get("current_level", "beginner"))
        progress.completed_exercises = data.get("completed_exercises", [])
        progress.mastered_letters = data.get("mastered_letters", [])
        progress.mastered_words = data.get("mastered_words", [])
        progress.total_practice_time = data.get("total_practice_time", 0.0)
        progress.streak_days = data.get("streak_days", 0)
        progress.last_practice_date = data.get("last_practice_date")
        
        # Reconstruct exercise results
        for result_data in data.get("exercise_results", []):
            result = ExerciseResult(
                exercise_id=result_data["exercise_id"],
                completed=result_data["completed"],
                attempts=result_data["attempts"],
                time_taken=result_data["time_taken"],
                accuracy=result_data["accuracy"],
                errors=result_data.get("errors", []),
                timestamp=result_data.get("timestamp", "")
            )
            progress.exercise_results.append(result)
        
        return progress


class LearningModeManager:
    """Manages interactive learning sessions."""
    
    def __init__(self, progress_file: str = "user_progress.json"):
        """
        Initialize learning mode manager.
        
        Args:
            progress_file: Path to save user progress
        """
        self.progress_file = Path(progress_file)
        self.exercises: List[Exercise] = []
        self.current_exercise: Optional[Exercise] = None
        self.user_progress: Optional[UserProgress] = None
        self.session_start_time: Optional[float] = None
        
        # Load default exercises
        self._create_default_exercises()
    
    def _create_default_exercises(self):
        """Create default exercise library."""
        # Alphabet exercises (Beginner)
        for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            self.exercises.append(Exercise(
                id=f"alphabet_{letter}",
                type=ExerciseType.ALPHABET_RECOGNITION,
                difficulty=DifficultyLevel.BEGINNER,
                title=f"Lettre {letter}",
                description=f"Pratiquez la lettre {letter} en langue des signes",
                target=letter,
                hints=[
                    f"Formez la lettre {letter} avec vos doigts",
                    "Maintenez la position pendant 2 secondes",
                    "Assurez-vous que la caméra voit bien votre main"
                ],
                required_repetitions=3
            ))
        
        # Common words (Intermediate)
        common_words = [
            "BONJOUR", "MERCI", "OUI", "NON", "SALUT",
            "AU REVOIR", "PARDON", "AIDE", "FAMILLE", "AMI"
        ]
        
        for word in common_words:
            self.exercises.append(Exercise(
                id=f"word_{word.lower().replace(' ', '_')}",
                type=ExerciseType.WORD_SPELLING,
                difficulty=DifficultyLevel.INTERMEDIATE,
                title=f"Mot : {word}",
                description=f"Épelez le mot '{word}' lettre par lettre",
                target=word,
                hints=[
                    f"Épeler : {' - '.join(word)}",
                    "Faites une pause entre chaque lettre",
                    "Soyez précis dans vos mouvements"
                ],
                time_limit=len(word) * 5,  # 5 seconds per letter
                required_repetitions=2
            ))
        
        # Phrases (Advanced)
        phrases = [
            "BONJOUR COMMENT ALLEZ VOUS",
            "MERCI BEAUCOUP",
            "BONNE JOURNEE",
            "A BIENTOT"
        ]
        
        for phrase in phrases:
            self.exercises.append(Exercise(
                id=f"phrase_{phrase.lower().replace(' ', '_')}",
                type=ExerciseType.PHRASE_PRACTICE,
                difficulty=DifficultyLevel.ADVANCED,
                title=f"Phrase : {phrase}",
                description=f"Signez la phrase complète : '{phrase}'",
                target=phrase,
                hints=[
                    "Séparez clairement chaque mot",
                    "Maintenez un rythme constant",
                    "Utilisez les expressions faciales appropriées"
                ],
                time_limit=len(phrase.split()) * 10,  # 10 seconds per word
                required_repetitions=1
            ))
        
        # Speed challenges (Expert)
        for i in range(5):
            letters = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=5))
            self.exercises.append(Exercise(
                id=f"speed_challenge_{i+1}",
                type=ExerciseType.SPEED_CHALLENGE,
                difficulty=DifficultyLevel.EXPERT,
                title=f"Défi Vitesse #{i+1}",
                description=f"Signez rapidement : {letters}",
                target=letters,
                hints=[
                    "Soyez rapide mais précis",
                    "Pas besoin de pause entre les lettres",
                    "Concentration maximale !"
                ],
                time_limit=15,
                required_repetitions=1
            ))
    
    def load_user_progress(self, user_id: str) -> UserProgress:
        """Load or create user progress."""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if data.get("user_id") == user_id:
                        self.user_progress = UserProgress.from_dict(data)
                        print(f"Loaded progress for user: {user_id}")
                        return self.user_progress
            except Exception as e:
                print(f"Error loading progress: {e}")
        
        # Create new progress
        self.user_progress = UserProgress(user_id=user_id)
        print(f"Created new progress for user: {user_id}")
        return self.user_progress
    
    def save_user_progress(self):
        """Save user progress to file."""
        if not self.user_progress:
            return
        
        try:
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                json.dump(self.user_progress.to_dict(), f, ensure_ascii=False, indent=2)
            print("Progress saved")
        except Exception as e:
            print(f"Error saving progress: {e}")
    
    def get_recommended_exercises(self, count: int = 5) -> List[Exercise]:
        """Get recommended exercises based on user level."""
        if not self.user_progress:
            return self.exercises[:count]
        
        # Filter by difficulty level
        level = self.user_progress.current_level
        suitable = [e for e in self.exercises if e.difficulty == level]
        
        # Exclude already completed
        uncompleted = [e for e in suitable if e.id not in self.user_progress.completed_exercises]
        
        # If all completed at this level, include some from next level
        if not uncompleted:
            next_level_map = {
                DifficultyLevel.BEGINNER: DifficultyLevel.INTERMEDIATE,
                DifficultyLevel.INTERMEDIATE: DifficultyLevel.ADVANCED,
                DifficultyLevel.ADVANCED: DifficultyLevel.EXPERT,
                DifficultyLevel.EXPERT: DifficultyLevel.EXPERT
            }
            next_level = next_level_map.get(level, level)
            suitable = [e for e in self.exercises if e.difficulty == next_level]
            uncompleted = suitable
        
        return uncompleted[:count]
    
    def start_exercise(self, exercise: Exercise):
        """Begin an exercise session."""
        self.current_exercise = exercise
        self.session_start_time = datetime.now().timestamp()
        print(f"\n{'='*60}")
        print(f"Exercise: {exercise.title}")
        print(f"Difficulty: {exercise.difficulty.value.upper()}")
        print(f"{'='*60}")
        print(f"\n{exercise.description}\n")
        
        if exercise.time_limit:
            print(f"⏱️  Time limit: {exercise.time_limit} seconds")
        print(f"🎯 Target: {exercise.target}")
        print(f"🔁 Repetitions required: {exercise.required_repetitions}\n")
        
        print("💡 Hints:")
        for i, hint in enumerate(exercise.hints, 1):
            print(f"   {i}. {hint}")
        print()
    
    def complete_exercise(
        self,
        success: bool,
        attempts: int,
        accuracy: float,
        errors: Optional[List[str]] = None
    ) -> ExerciseResult:
        """Record exercise completion."""
        if not self.current_exercise or not self.session_start_time:
            raise ValueError("No active exercise")
        
        time_taken = datetime.now().timestamp() - self.session_start_time
        
        result = ExerciseResult(
            exercise_id=self.current_exercise.id,
            completed=success,
            attempts=attempts,
            time_taken=time_taken,
            accuracy=accuracy,
            errors=errors or []
        )
        
        if self.user_progress:
            self.user_progress.add_result(result)
            
            # Check for level up
            if self.user_progress.should_level_up():
                self._level_up()
            
            self.save_user_progress()
        
        self._print_result(result)
        
        self.current_exercise = None
        self.session_start_time = None
        
        return result
    
    def _level_up(self):
        """Advance user to next difficulty level."""
        if not self.user_progress:
            return
        
        level_order = [
            DifficultyLevel.BEGINNER,
            DifficultyLevel.INTERMEDIATE,
            DifficultyLevel.ADVANCED,
            DifficultyLevel.EXPERT
        ]
        
        current_idx = level_order.index(self.user_progress.current_level)
        if current_idx < len(level_order) - 1:
            new_level = level_order[current_idx + 1]
            self.user_progress.current_level = new_level
            print(f"\n🎉 LEVEL UP! New level: {new_level.value.upper()} 🎉\n")
    
    def _print_result(self, result: ExerciseResult):
        """Print exercise results."""
        print(f"\n{'='*60}")
        print(f"Results")
        print(f"{'='*60}\n")
        
        if result.completed:
            print("✅ Exercise completed successfully!")
        else:
            print("❌ Exercise not completed")
        
        print(f"\n📊 Statistics:")
        print(f"   Attempts: {result.attempts}")
        print(f"   Time: {result.time_taken:.1f} seconds")
        print(f"   Accuracy: {result.accuracy*100:.1f}%")
        
        if result.errors:
            print(f"\n⚠️  Errors detected:")
            for error in result.errors:
                print(f"   - {error}")
        
        print()
    
    def get_progress_summary(self) -> dict:
        """Get user progress summary."""
        if not self.user_progress:
            return {}
        
        return {
            "level": self.user_progress.current_level.value,
            "completed_exercises": len(self.user_progress.completed_exercises),
            "total_exercises": len(self.exercises),
            "completion_rate": len(self.user_progress.completed_exercises) / len(self.exercises) * 100,
            "average_accuracy": self.user_progress.get_accuracy_trend(20),
            "practice_time_hours": self.user_progress.total_practice_time / 60,
            "streak_days": self.user_progress.streak_days,
            "mastered_letters": len(self.user_progress.mastered_letters),
            "mastered_words": len(self.user_progress.mastered_words)
        }


if __name__ == "__main__":
    # Demo usage
    print("Interactive Learning Mode Demo")
    print("=" * 60)
    
    # Create learning manager
    manager = LearningModeManager()
    
    # Load user progress
    progress = manager.load_user_progress("demo_user")
    
    # Get recommended exercises
    print("\nRecommended Exercises:")
    recommended = manager.get_recommended_exercises(3)
    for i, ex in enumerate(recommended, 1):
        print(f"{i}. {ex.title} ({ex.difficulty.value})")
    
    # Start an exercise
    print()
    manager.start_exercise(recommended[0])
    
    # Simulate completion
    import time
    time.sleep(2)
    result = manager.complete_exercise(
        success=True,
        attempts=1,
        accuracy=0.95,
        errors=[]
    )
    
    # Show progress summary
    summary = manager.get_progress_summary()
    print("\n📈 Progress Summary:")
    for key, value in summary.items():
        print(f"   {key}: {value}")
