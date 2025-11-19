"""Enhanced data collection tool with extended alphabet support.

Collects hand landmark data for training ML models on A-Z sign language letters.
Supports extended alphabet and provides real-time feedback.
"""

import cv2
import mediapipe as mp
import pandas as pd
import argparse
import time
from pathlib import Path
from typing import Optional

# Try to import extended alphabet detector
try:
    from letters_conditions_extended import detect_letter_extended
    HAS_EXTENDED = True
except ImportError:
    HAS_EXTENDED = False
    print("Warning: letters_conditions_extended not found. Using basic detection only.")


class EnhancedDataCollector:
    """Enhanced data collector with A-Z support and real-time heuristic detection."""
    
    def __init__(self, output_path: str = "machine_learning/data_extended.csv"):
        """
        Initialize data collector.
        
        Args:
            output_path: Path to save collected data CSV
        """
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Data storage
        self.samples = []
        self.target_samples = 100  # Samples per letter
        
        # UI state
        self.current_letter = None
        self.letter_counts = {chr(i): 0 for i in range(ord('A'), ord('Z') + 1)}
        self.is_recording = False
        self.last_sample_time = 0
        self.sample_cooldown = 0.2  # Seconds between samples
    
    def extract_features(self, hand_landmarks) -> list:
        """
        Extract 63-dimensional feature vector from hand landmarks.
        
        Args:
            hand_landmarks: MediaPipe hand landmarks
            
        Returns:
            List of 63 float values (21 landmarks × 3 coordinates)
        """
        features = []
        for landmark in hand_landmarks.landmark:
            features.extend([landmark.x, landmark.y, landmark.z])
        return features
    
    def collect_for_letter(self, letter: str, cap: cv2.VideoCapture) -> int:
        """
        Collect samples for a specific letter.
        
        Args:
            letter: Target letter (A-Z)
            cap: OpenCV video capture
            
        Returns:
            Number of samples collected
        """
        self.current_letter = letter.upper()
        samples_collected = 0
        
        print(f"\n{'='*60}")
        print(f"Collecting data for letter: {self.current_letter}")
        print(f"Target: {self.target_samples} samples")
        print(f"{'='*60}")
        print("\nControls:")
        print("  SPACE : Start/Stop recording")
        print("  Q     : Skip to next letter")
        print("  ESC   : Exit collection")
        print("\nPosition your hand to form the letter and press SPACE to start recording.")
        
        while samples_collected < self.target_samples:
            success, frame = cap.read()
            if not success:
                break
            
            # Mirror frame for natural interaction
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process hand detection
            results = self.hands.process(frame_rgb)
            
            # Detection status
            detected_letter = None
            
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                
                # Draw landmarks
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )
                
                # Heuristic detection feedback
                if HAS_EXTENDED:
                    detected_letter = detect_letter_extended(hand_landmarks)
                
                # Record sample if recording is active
                if self.is_recording and results.multi_hand_landmarks:
                    current_time = time.time()
                    if current_time - self.last_sample_time >= self.sample_cooldown:
                        features = self.extract_features(hand_landmarks)
                        self.samples.append({
                            'label': self.current_letter,
                            'features': features
                        })
                        samples_collected += 1
                        self.letter_counts[self.current_letter] += 1
                        self.last_sample_time = current_time
            
            # Draw UI overlay
            self._draw_ui(frame, samples_collected, detected_letter)
            
            # Display frame
            cv2.imshow('Enhanced Data Collection', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):  # Toggle recording
                self.is_recording = not self.is_recording
                status = "STARTED" if self.is_recording else "STOPPED"
                print(f"Recording {status}")
            
            elif key == ord('q') or key == ord('Q'):  # Skip letter
                print(f"Skipping letter {self.current_letter} (collected {samples_collected} samples)")
                break
            
            elif key == 27:  # ESC - exit
                print("Collection interrupted by user")
                return samples_collected
        
        self.is_recording = False
        print(f"\n✓ Completed {self.current_letter}: {samples_collected} samples collected")
        
        return samples_collected
    
    def _draw_ui(self, frame, samples_collected: int, detected_letter: Optional[str]):
        """Draw UI overlay on frame."""
        height, width = frame.shape[:2]
        
        # Status bar background
        cv2.rectangle(frame, (0, 0), (width, 120), (0, 0, 0), -1)
        
        # Current letter and progress
        text = f"Target Letter: {self.current_letter}"
        cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        
        text = f"Progress: {samples_collected}/{self.target_samples}"
        cv2.putText(frame, text, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Recording status
        if self.is_recording:
            cv2.circle(frame, (width - 40, 40), 15, (0, 0, 255), -1)
            cv2.putText(frame, "REC", (width - 80, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Heuristic detection feedback
        if detected_letter:
            color = (0, 255, 0) if detected_letter == self.current_letter else (0, 165, 255)
            text = f"Detected: {detected_letter}"
            cv2.putText(frame, text, (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Progress bar
        progress_width = int((samples_collected / self.target_samples) * (width - 40))
        cv2.rectangle(frame, (20, height - 30), (20 + progress_width, height - 10), (0, 255, 0), -1)
        cv2.rectangle(frame, (20, height - 30), (width - 20, height - 10), (255, 255, 255), 2)
    
    def save_data(self):
        """Save collected data to CSV."""
        if not self.samples:
            print("No samples to save")
            return
        
        # Prepare DataFrame
        data = []
        for sample in self.samples:
            row = {'label': sample['label']}
            for i, value in enumerate(sample['features']):
                row[f'feature_{i}'] = value
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Append or create new file
        if self.output_path.exists():
            existing_df = pd.read_csv(self.output_path)
            df = pd.concat([existing_df, df], ignore_index=True)
        
        df.to_csv(self.output_path, index=False)
        
        print(f"\n{'='*60}")
        print(f"✓ Data saved to: {self.output_path}")
        print(f"Total samples: {len(df)}")
        print(f"\nSamples per letter:")
        print(df['label'].value_counts().sort_index())
        print(f"{'='*60}")
    
    def collect_alphabet(self, letters: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ", samples_per_letter: int = 100):
        """
        Collect data for multiple letters.
        
        Args:
            letters: String of letters to collect
            samples_per_letter: Target samples per letter
        """
        self.target_samples = samples_per_letter
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        print("\n" + "="*60)
        print("ENHANCED DATA COLLECTION - A-Z ALPHABET")
        print("="*60)
        if HAS_EXTENDED:
            print("✓ Extended alphabet detection enabled (A-Z)")
        else:
            print("⚠ Basic detection only (extended detector not found)")
        print(f"\nCollecting {samples_per_letter} samples for each of: {letters}")
        print("="*60)
        
        try:
            for letter in letters:
                self.collect_for_letter(letter, cap)
        
        except KeyboardInterrupt:
            print("\n\nCollection interrupted by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.save_data()
            
            # Summary
            print("\n" + "="*60)
            print("COLLECTION SUMMARY")
            print("="*60)
            print(f"Total samples collected: {len(self.samples)}")
            for letter, count in sorted(self.letter_counts.items()):
                if count > 0:
                    print(f"  {letter}: {count} samples")
            print("="*60)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Enhanced data collection for sign language detection (A-Z)"
    )
    parser.add_argument(
        '--letters',
        type=str,
        default='ABCDEFGHIJKLMNOPQRSTUVWXYZ',
        help='Letters to collect (default: full alphabet A-Z)'
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=100,
        help='Samples per letter (default: 100)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='machine_learning/data_extended.csv',
        help='Output CSV file path'
    )
    
    args = parser.parse_args()
    
    collector = EnhancedDataCollector(output_path=args.output)
    collector.collect_alphabet(
        letters=args.letters.upper(),
        samples_per_letter=args.samples
    )


if __name__ == "__main__":
    main()
