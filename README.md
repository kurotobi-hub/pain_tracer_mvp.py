# pain_tracer_mvp.py"""
"""
Filename: pain_nexus_mvp.py
Author: kurotobi
Version: 2.0 (Multimodal Fusion)
Description: 
    A unified prototype integrating Facial Analysis (Sensory/Emotional) 
    and Verbal Analysis (Cognitive/Evaluative) to assess pain
    according to the multidimensional IASP 2020 definition.
"""

import cv2
import mediapipe as mp
import numpy as np
import re
import time
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

# --- Data Structures ---
@dataclass
class PainState:
    visual_score: float  # 0.0 - 10.0
    cognitive_score: float  # 0.0 - 10.0
    combined_score: float  # 0.0 - 10.0
    status_label: str
    cognitive_context: str

# --- Module 1: Cognitive Analyzer (The "Mind") ---
class CognitiveEncoder:
    """
    Parses verbal reports for 'Catastrophizing' patterns.
    Reflects the 'Learned Concept' aspect of pain.
    """
    def __init__(self):
        # English markers based on Pain Catastrophizing Scale (PCS)
        self.markers = {
            'rumination': [
                r"can'?t stop", r"always", r"never end", r"worry", r"thinking about", 
                r"constant"
            ],
            'helplessness': [
                r"nothing i can do", r"useless", r"hopeless", r"overwhelming", 
                r"can'?t go on", r"impossible"
            ],
            'magnification': [
                r"unbearable", r"killing me", r"terrible", r"worst", r"explode", 
                r"torture", r"agony"
            ]
        }

    def analyze(self, text: str) -> Tuple[float, str]:
        if not text:
            return 0.0, "No verbal report"

        hits = 0
        details = []
        
        # Linear scan for MVP (O(n))
        for category, patterns in self.markers.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    hits += 1
                    details.append(category.upper())
        
        # Logarithmic scoring to prevent runaway values
        score = min(10.0, hits * 3.0)
        context = f"Detected: {', '.join(set(details))}" if details else "Adaptive coping"
        return score, context

# --- Module 2: Visual Analyzer (The "Body") ---
class VisualEncoder:
    """
    Tracks facial grimacing (Brow Lowering + Orbital Tightening).
    Reflects the 'Sensory/Emotional' aspect of pain.
    """
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.baseline = None
        self.alpha = 0.7  # Smoothing factor

    def _dist(self, p1, p2, w, h):
        return np.sqrt(((p1.x - p2.x) * w)**2 + ((p1.y - p2.y) * h)**2)

    def process(self, image) -> Tuple[float, object]:
        h, w, _ = image.shape
        rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_img)
        
        raw_score = 0.0
        
        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            
            # 1. Brow Lowering (Corrugator supercilii): Point 9 (Mid-Brow) <-> Point 6 (Nose Bridge)
            brow_dist = self._dist(lm[9], lm[6], w, h)
            
            # 2. Orbital Tightening (Squint): Point 159 (Top Lid) <-> Point 145 (Bottom Lid)
            eye_dist = self._dist(lm[159], lm[145], w, h)
            
            current_val = brow_dist + eye_dist

            if self.baseline is None:
                self.baseline = current_val
            
            # Grimace = Contraction (Distance decreases)
            delta = self.baseline - current_val
            
            # Normalize to 0-10 (Heuristic calibration)
            raw_score = max(0.0, min(10.0, (delta / (self.baseline + 1e-6)) * 150))
            
            return raw_score, results.multi_face_landmarks
        
        return 0.0, None

# --- Module 3: System Nexus (The Controller) ---
class PainAssessmentNexus:
    def __init__(self):
        self.visual = VisualEncoder()
        self.cognitive = CognitiveEncoder()
        
        # Config
        self.WEIGHT_VISUAL = 0.4
        self.WEIGHT_COGNITIVE = 0.6 # Cognitive bias often dominates chronic pain
        
        # State
        self.current_text = ""
        self.last_cognitive_score = 0.0
        self.last_cognitive_ctx = "Waiting for report..."

    def update_verbal_input(self, text: str):
        self.current_text = text
        score, ctx = self.cognitive.analyze(text)
        self.last_cognitive_score = score
        self.last_cognitive_ctx = ctx

    def run(self):
        cap = cv2.VideoCapture(0)
        print("--- PainAssessmentNexus Active ---")
        print("KEYS: [1] Mild Report  [2] High Distress  [3] Catastrophizing  [Q] Quit")
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success: break
            
            # 1. Visual Processing
            v_score, landmarks = self.visual.process(frame)
            
            # 2. Data Fusion
            combined = (v_score * self.WEIGHT_VISUAL) + \
                       (self.last_cognitive_score * self.WEIGHT_COGNITIVE)
            
            # 3. Visualization
            self._draw_ui(frame, v_score, self.last_cognitive_score, combined, landmarks)
            
            cv2.imshow('IASP Pain Nexus (Snakajima Build)', frame)
            
            # User Input Simulation (Simulating Voice-to-Text)
            key = cv2.waitKey(5) & 0xFF
            if key == ord('q'): break
            elif key == ord('1'): self.update_verbal_input("It hurts a little, but I can manage.")
            elif key == ord('2'): self.update_verbal_input("It's really painful, I can't work today.")
            elif key == ord('3'): self.update_verbal_input("It never ends! It's killing me! It's hopeless!")
            elif key == ord('c'): self.visual.baseline = None # Recalibrate Visual

        cap.release()
        cv2.destroyAllWindows()

    def _draw_ui(self, img, v_score, c_score, total, landmarks):
        # Dark overlay for dashboard
        cv2.rectangle(img, (0, 0), (400, 180), (20, 20, 20), -1)
        
        # Colors
        color = (0, 255, 0) if total < 4 else (0, 165, 255) if total < 7 else (0, 0, 255)
        
        # Text Rendering Helper
        def put(txt, y, scale=0.6, col=(200,200,200)):
            cv2.putText(img, txt, (15, y), cv2.FONT_HERSHEY_SIMPLEX, scale, col, 1)

        put(f"TOTAL PAIN INDEX: {total:.1f} / 10", 30, 0.8, color)
        put(f"--------------------------------", 45)
        put(f"1. VISUAL (Grimace):   {v_score:.1f}", 65)
        put(f"2. COGNITIVE (Verbal): {c_score:.1f}", 85)
        put(f"   Context: {self.last_cognitive_ctx}", 105, 0.5, (150,150,150))
        put(f"   Input: \"{self.current_text[:35]}...\"", 125, 0.4, (100,255,255))
        put(f"--------------------------------", 140)
        put(f"Mode: Multimodal Fusion (IASP 2020)", 160, 0.5, (100,100,100))

        # Draw Face Mesh (Subtle)
        if landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                image=img,
                landmark_list=landmarks[0],
                connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1)
            )

if __name__ == "__main__":
    app = PainAssessmentNexus()
    app.run()
