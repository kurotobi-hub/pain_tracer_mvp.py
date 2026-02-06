# pain_tracer_mvp.py"""
Filename: pain_tracer_mvp.py
Author: kurotobi
Description: 
    A rapid prototype to quantify 'Pain-associated behaviors'  
    via facial landmark analysis (Grimace Scale approximation).
    Focuses on 'unpleasant sensory and emotional experience' 
    manifested in brow lowering and orbital tightening.
"""

import cv2
import mediapipe as mp
import numpy as np
import time

class PainTracer:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
        
        # Color definitions for UI
        self.COLOR_OK = (0, 255, 0)
        self.COLOR_WARN = (0, 165, 255)
        self.COLOR_CRITICAL = (0, 0, 255)

    def calculate_distance(self, p1, p2, img_w, img_h):
        """Euclidean distance between two normalized points"""
        x1, y1 = p1.x * img_w, p1.y * img_h
        x2, y2 = p2.x * img_w, p2.y * img_h
        return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Camera not found.")
            return

        print("--- PainTracer MVP Started ---")
        print("Press 'q' to quit.")

        # Calibration baselines (Dynamic adjustment recommended for production)
        baseline_brow_eye = None
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break

            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            img_h, img_w, _ = image.shape

            pain_score = 0.0
            status_text = "Calibrating..."
            status_color = self.COLOR_OK

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Key Landmarks for Pain Expression (Primal Sound & Face)
                    # 1. Brow Lowering (Corrugator supercilii): 
                    #    Distance between mid-brow (10) and nose bridge (6) or inner eyes
                    # 2. Orbital Tightening (Orbicularis oculi):
                    #    Vertical eye opening
                    
                    # Indices from MediaPipe Mesh
                    pt_brow_mid = face_landmarks.landmark[9]
                    pt_nose_top = face_landmarks.landmark[6]
                    pt_eye_L_top = face_landmarks.landmark[159]
                    pt_eye_L_bot = face_landmarks.landmark[145]
                    
                    # Calculate metrics
                    brow_nose_dist = self.calculate_distance(pt_brow_mid, pt_nose_top, img_w, img_h)
                    eye_open_dist = self.calculate_distance(pt_eye_L_top, pt_eye_L_bot, img_w, img_h)

                    # Simple heuristic: Narrowing eyes + Lowering brows = Pain/Discomfort
                    # Note: Pain is biological, psychological, and social, 
                    # so this is just the biological signal layer.
                    
                    if baseline_brow_eye is None:
                         # Initialize baseline (Assume neutral face at start - highly simplified)
                         baseline_brow_eye = brow_nose_dist + eye_open_dist
                    
                    current_val = brow_nose_dist + eye_open_dist
                    
                    # Calculate deviation (pain often involves contraction)
                    deviation = baseline_brow_eye - current_val
                    
                    # Normalize roughly to 0-10 scale
                    pain_score = max(0, min(10, (deviation / baseline_brow_eye) * 100))

                    # Visual Feedback
                    if pain_score > 6:
                        status_text = "High Discomfort Detected"
                        status_color = self.COLOR_CRITICAL
                    elif pain_score > 3:
                        status_text = "Mild Discomfort"
                        status_color = self.COLOR_WARN
                    else:
                        status_text = "Neutral / Relaxed"
                        status_color = self.COLOR_OK

                    # Draw landmarks for debug
                    cv2.circle(image, (int(pt_brow_mid.x * img_w), int(pt_brow_mid.y * img_h)), 3, (0,255,255), -1)

            # Dashboard Overlay
            cv2.rectangle(image, (10, 10), (350, 120), (0, 0, 0), -1)
            cv2.putText(image, f"Pain Signal (AI-Est): {pain_score:.1f}/10", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(image, f"Status: {status_text}", (20, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            
            # Context Note: "Verbal description is only one of several behaviors" 
            cv2.putText(image, "IASP 2020 Definition Compliant Mode", (20, 105), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

            cv2.imshow('PainTracer MVP - Snakajima Build', image)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = PainTracer()
    app.run()
