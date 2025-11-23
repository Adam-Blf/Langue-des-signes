import customtkinter as ctk
import cv2
from PIL import Image, ImageTk
import threading
from lsf_model import LSFDetector
from letters_conditions import detect_letter_rules
import time
import datetime
import os
import csv
import numpy as np
import pickle
import subprocess

class LSFApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Window setup
        self.title("LSF Detector - Pro Edition")
        self.geometry("1280x850")
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")

        # Data
        self.detector = LSFDetector()
        self.cap = None
        self.is_running = False
        self.current_camera_index = 0
        self.recording = False
        self.collecting_data = False
        self.video_writer = None
        self.start_time = 0
        self.model = None
        self.current_label = ""
        self.data_file = "machine_learning/data.csv"
        
        # Check data file compatibility
        self.check_data_file()
        
        # Load Model
        self.load_model()

        # Layout
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._create_sidebar()
        self._create_main_area()

    def check_data_file(self):
        # Check if data file exists and has correct dimension
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r') as f:
                    reader = csv.reader(f)
                    row = next(reader, None)
                    if row:
                        # Expected: 63 features + 1 label = 64 columns
                        if len(row) != 64:
                            print(f"Warning: Data file has {len(row)} columns, expected 64. Renaming old file.")
                            f.close()
                            os.rename(self.data_file, self.data_file + ".bak")
            except Exception as e:
                print(f"Error checking data file: {e}")

    def load_model(self):
        model_path = "machine_learning/model.p"
        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    data = pickle.load(f)
                    self.model = data['model']
                    model_type = data.get('type', 'Unknown')
                    acc = data.get('accuracy', 0) * 100
                print(f"Model loaded successfully: {model_type} ({acc:.1f}%)")
                self.status_label.configure(text=f"● Model: {model_type}", text_color="#2CC985")
            except Exception as e:
                print(f"Error loading model: {e}")
                self.model = None
        else:
            print("No model found.")
            self.model = None

    def _create_sidebar(self):
        self.sidebar_frame = ctk.CTkFrame(self, width=240, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(9, weight=1)

        # Logo
        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="LSF STUDIO", font=ctk.CTkFont(size=28, weight="bold", family="Roboto"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(30, 20))

        # Tabs for Mode
        self.mode_tab = ctk.CTkTabview(self.sidebar_frame, width=200, height=100)
        self.mode_tab.grid(row=1, column=0, padx=10, pady=10)
        self.mode_tab.add("Live")
        self.mode_tab.add("Train")
        self.mode_tab.set("Live")

        # Camera Controls
        self.control_label = ctk.CTkLabel(self.sidebar_frame, text="CONTROLS", font=ctk.CTkFont(size=12, weight="bold"), text_color="gray")
        self.control_label.grid(row=2, column=0, padx=20, pady=(10, 5), sticky="w")

        self.start_button = ctk.CTkButton(self.sidebar_frame, text="Start Camera", command=self.toggle_camera, height=40, font=ctk.CTkFont(weight="bold"), fg_color="#2CC985", hover_color="#229A65", text_color="white")
        self.start_button.grid(row=3, column=0, padx=20, pady=10)

        self.camera_selector = ctk.CTkOptionMenu(self.sidebar_frame, values=["Camera 0", "Camera 1", "Camera 2"], command=self.change_camera)
        self.camera_selector.grid(row=4, column=0, padx=20, pady=10)

        # Tools
        self.tools_label = ctk.CTkLabel(self.sidebar_frame, text="TOOLS", font=ctk.CTkFont(size=12, weight="bold"), text_color="gray")
        self.tools_label.grid(row=5, column=0, padx=20, pady=(20, 5), sticky="w")

        self.snapshot_btn = ctk.CTkButton(self.sidebar_frame, text="📸 Snapshot", command=self.take_snapshot, fg_color="transparent", border_width=2, text_color=("gray10", "#DCE4EE"))
        self.snapshot_btn.grid(row=6, column=0, padx=20, pady=10)
        
        # Training Controls
        self.train_label_entry = ctk.CTkEntry(self.sidebar_frame, placeholder_text="Sign Label (e.g. Hello)")
        self.train_label_entry.grid(row=7, column=0, padx=20, pady=(10, 5))

        self.collect_btn = ctk.CTkButton(self.sidebar_frame, text="🔴 Collect Data", command=self.toggle_data_collection, fg_color="#E53935", hover_color="#C62828", state="disabled")
        self.collect_btn.grid(row=8, column=0, padx=20, pady=5)

        self.train_btn = ctk.CTkButton(self.sidebar_frame, text="🧠 Train Model", command=self.run_training, fg_color="#1E88E5", hover_color="#1565C0")
        self.train_btn.grid(row=9, column=0, padx=20, pady=5)

        # Settings
        self.settings_label = ctk.CTkLabel(self.sidebar_frame, text="SETTINGS", font=ctk.CTkFont(size=12, weight="bold"), text_color="gray")
        self.settings_label.grid(row=10, column=0, padx=20, pady=(20, 5), sticky="w")

        self.draw_landmarks_var = ctk.BooleanVar(value=True)
        self.draw_landmarks_switch = ctk.CTkSwitch(self.sidebar_frame, text="Show AI Skeleton", variable=self.draw_landmarks_var)
        self.draw_landmarks_switch.grid(row=11, column=0, padx=20, pady=10, sticky="w")

        # Theme
        self.theme_mode = ctk.StringVar(value="Dark")
        self.theme_switch = ctk.CTkSwitch(self.sidebar_frame, text="Dark Mode", command=self.toggle_theme, variable=self.theme_mode, onvalue="Dark", offvalue="Light")
        self.theme_switch.select()
        self.theme_switch.grid(row=12, column=0, padx=20, pady=20, sticky="s")

    def _create_main_area(self):
        self.main_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)

        # Video Container with Border
        self.video_border = ctk.CTkFrame(self.main_frame, corner_radius=16, fg_color="#333333", border_width=2, border_color="#444444")
        self.video_border.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        self.video_border.grid_rowconfigure(0, weight=1)
        self.video_border.grid_columnconfigure(0, weight=1)

        self.video_label = ctk.CTkLabel(self.video_border, text="Camera Offline\n\nClick 'Start Camera' to begin", font=ctk.CTkFont(size=20), text_color="gray")
        self.video_label.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)

        # Info Panel
        self.info_panel = ctk.CTkFrame(self.main_frame, height=180, corner_radius=15, fg_color=("gray90", "gray16"))
        self.info_panel.grid(row=1, column=0, sticky="ew", padx=0, pady=(20, 0))
        self.info_panel.grid_columnconfigure((0, 1, 2), weight=1)

        # Prediction Box
        self.pred_box = ctk.CTkFrame(self.info_panel, fg_color="transparent")
        self.pred_box.grid(row=0, column=1, pady=20)
        
        ctk.CTkLabel(self.pred_box, text="DETECTED SIGN", font=ctk.CTkFont(size=12, weight="bold"), text_color="gray").pack()
        self.prediction_text = ctk.CTkLabel(self.pred_box, text="---", font=ctk.CTkFont(size=48, weight="bold"), text_color="#2CC985")
        self.prediction_text.pack()

        # Confidence Box
        self.conf_box = ctk.CTkFrame(self.info_panel, fg_color="transparent")
        self.conf_box.grid(row=0, column=2, pady=20, padx=20, sticky="e")
        
        ctk.CTkLabel(self.conf_box, text="CONFIDENCE", font=ctk.CTkFont(size=12, weight="bold"), text_color="gray").pack(anchor="e")
        self.confidence_bar = ctk.CTkProgressBar(self.conf_box, width=200, height=15, progress_color="#2CC985")
        self.confidence_bar.set(0)
        self.confidence_bar.pack(pady=5)
        self.confidence_text = ctk.CTkLabel(self.conf_box, text="0%", font=ctk.CTkFont(size=12), text_color="gray")
        self.confidence_text.pack(anchor="e")

        # Status Box
        self.status_box = ctk.CTkFrame(self.info_panel, fg_color="transparent")
        self.status_box.grid(row=0, column=0, pady=20, padx=20, sticky="w")
        self.status_label = ctk.CTkLabel(self.status_box, text="● Ready", font=ctk.CTkFont(size=14), text_color="gray")
        self.status_label.pack(anchor="w")
        
        self.fps_label = ctk.CTkLabel(self.status_box, text="FPS: 0", font=ctk.CTkFont(size=12), text_color="gray")
        self.fps_label.pack(anchor="w")

    def toggle_theme(self):
        ctk.set_appearance_mode(self.theme_mode.get())

    def toggle_data_collection(self):
        if not self.collecting_data:
            label = self.train_label_entry.get()
            if not label:
                self.status_label.configure(text="⚠ Enter a label first!", text_color="orange")
                return
            
            self.current_label = label
            self.collecting_data = True
            self.collect_btn.configure(text="⏹ Stop Collection", fg_color="#B71C1C")
            self.status_label.configure(text=f"● Collecting '{label}'...", text_color="#E53935")
            
            # Initialize CSV if needed
            if not os.path.exists(self.data_file):
                with open(self.data_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    # We don't know the exact number of landmarks yet, but we can just write data rows
                    # Or we can write a header later. For simplicity, let's just write data.
                    pass
        else:
            self.collecting_data = False
            self.collect_btn.configure(text="🔴 Collect Data", fg_color="#E53935")
            self.status_label.configure(text="● Live", text_color="#2CC985")
            self.train_label_entry.delete(0, 'end')

    def run_training(self):
        self.status_label.configure(text="⏳ Training Model...", text_color="yellow")
        self.update() # Force update UI
        
        try:
            # Run training script
            result = subprocess.run(["python", "machine_learning/train_model.py"], capture_output=True, text=True)
            print(result.stdout)
            if result.returncode == 0:
                self.status_label.configure(text="✅ Training Complete!", text_color="#2CC985")
                self.load_model() # Reload the new model
            else:
                self.status_label.configure(text="❌ Training Failed", text_color="red")
                print(result.stderr)
        except Exception as e:
            self.status_label.configure(text=f"❌ Error: {e}", text_color="red")

    def take_snapshot(self):
        if hasattr(self, 'last_frame') and self.last_frame is not None:
            if not os.path.exists("snapshots"):
                os.makedirs("snapshots")
            filename = f"snapshots/lsf_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            # Convert RGB back to BGR for saving with OpenCV
            save_img = cv2.cvtColor(self.last_frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filename, save_img)
            self.status_label.configure(text=f"Saved: {filename}", text_color="#2CC985")
            self.after(2000, lambda: self.status_label.configure(text="● Live", text_color="#2CC985"))

    def toggle_camera(self):
        if self.is_running:
            self.stop_camera()
        else:
            self.start_camera()

    def start_camera(self):
        self.cap = cv2.VideoCapture(self.current_camera_index)
        if not self.cap.isOpened():
            self.status_label.configure(text="● Error: No Camera", text_color="#EF5350")
            return
        
        self.is_running = True
        self.start_button.configure(text="Stop Camera", fg_color="#D32F2F", hover_color="#B71C1C")
        self.collect_btn.configure(state="normal")
        self.status_label.configure(text="● Live", text_color="#2CC985")
        self.video_label.configure(text="") # Clear offline text
        
        self.thread = threading.Thread(target=self.video_loop, daemon=True)
        self.thread.start()

    def stop_camera(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
        self.start_button.configure(text="Start Camera", fg_color="#2CC985", hover_color="#229A65")
        self.collect_btn.configure(state="disabled")
        self.status_label.configure(text="● Offline", text_color="gray")
        self.video_label.configure(image=None, text="Camera Offline")
        self.fps_label.configure(text="FPS: 0")
        
        if self.collecting_data:
            self.toggle_data_collection() # Stop recording if camera stops

    def change_camera(self, choice):
        self.current_camera_index = int(choice.split(" ")[1])
        if self.is_running:
            self.stop_camera()
            self.start_camera()

    def video_loop(self):
        prev_time = 0
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Calculate FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
            prev_time = curr_time

            # 1. Mirror Effect (Flip horizontally)
            frame = cv2.flip(frame, 1)
            self.last_frame = frame.copy() # Save for snapshot

            # Process frame
            results, image = self.detector.process_frame(frame)

            # Extract Keypoints (63-dim vector)
            keypoints = self.detector.extract_keypoints(results)
            
            # Rule-based Detection
            detected_letter = None
            method = ""
            
            if results and results.multi_hand_landmarks:
                detected_letter = detect_letter_rules(results.multi_hand_landmarks[0])
                if detected_letter:
                    method = "Rule-Based"

            # Data Collection
            if self.collecting_data:
                try:
                    # Append to CSV (Label first, then 63 features) - Matches reference repo format
                    row = [self.current_label] + list(keypoints)
                    with open(self.data_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(row)
                except Exception as e:
                    print(f"Error saving data: {e}")

            # Prediction (ML Fallback)
            if self.model and not self.collecting_data and not detected_letter:
                try:
                    # Reshape for prediction (1, -1)
                    # Check if keypoints are not all zeros (meaning hand detected)
                    if np.any(keypoints):
                        prediction = self.model.predict([keypoints])[0]
                        probs = self.model.predict_proba([keypoints])[0]
                        confidence = np.max(probs)

                        # Update UI
                        if confidence > 0.6: # Threshold
                            detected_letter = prediction
                            method = f"ML ({int(confidence * 100)}%)"
                            self.confidence_bar.set(confidence)
                            self.confidence_text.configure(text=f"{int(confidence * 100)}%")
                        else:
                            self.confidence_bar.set(0)
                            self.confidence_text.configure(text="0%")
                except Exception as e:
                    # print(f"Prediction error: {e}")
                    pass
            
            # Update UI with result
            if detected_letter:
                self.prediction_text.configure(text=detected_letter)
                self.status_label.configure(text=f"● Detected: {method}", text_color="#2CC985")
            elif not self.collecting_data:
                self.prediction_text.configure(text="...")
                self.status_label.configure(text="● Live", text_color="#2CC985")

            # Draw landmarks if enabled

            # Draw landmarks if enabled
            if self.draw_landmarks_var.get():
                image = self.detector.draw_landmarks(image, results)

            # Convert to PIL Image for CustomTkinter
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(image)
            
            # Resize to fit window (maintain aspect ratio)
            display_w = self.video_label.winfo_width()
            display_h = self.video_label.winfo_height()
            
            if display_w > 10 and display_h > 10: # Ensure valid dimensions
                img_ratio = img.width / img.height
                target_ratio = display_w / display_h
                
                if target_ratio > img_ratio:
                    new_h = display_h
                    new_w = int(new_h * img_ratio)
                else:
                    new_w = display_w
                    new_h = int(new_w / img_ratio)
                    
                img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

            ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(new_w, new_h))

            # Update GUI (must be on main thread, but ctk handles this reasonably well)
            try:
                self.video_label.configure(image=ctk_img)
                self.video_label.image = ctk_img # Keep reference
                
                # Update FPS every 10 frames or so to avoid flicker? 
                # Actually ctk label update is fast enough usually.
                if int(curr_time * 10) % 5 == 0: # Update roughly every 0.5s
                     self.fps_label.configure(text=f"FPS: {int(fps)}")
            except Exception:
                pass

            time.sleep(0.01)

if __name__ == "__main__":
    app = LSFApp()
    app.mainloop()
