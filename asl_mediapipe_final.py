#!/usr/bin/env python3
"""
ASL Hand Gesture Recognition with MediaPipe - Real-time Webcam Integration
Adapted for your trained models (ResNet50 Teacher + MobileNetV2 Student)
"""

import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import numpy as np
import time
import os
import queue
import threading

# ============================================================================
# CONFIGURATION - UPDATE THESE PATHS
# ============================================================================

TEACHER_MODEL_PATH = "./model_weights/teacher_best_aug.pth"
STUDENT_MODEL_PATH = "./model_weights/student_best_aug.pth"

CLASSES = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
NUM_CLASSES = 26
CONFIDENCE_THRESHOLD = 0.5

# ============================================================================
# TEXT TO SPEECH SETTINGS
# ============================================================================
import queue
import subprocess

SPEAK_INTERVAL = 3  # seconds
speech_queue = queue.Queue()
last_speak_time = 0

def tts_worker():
    while True:
        letter = speech_queue.get()
        if letter is None:
            break
        try:
            # Uses Windows built-in TTS - no pyttsx3 engine issues
            subprocess.run(
                ["powershell", "-Command",
                 f'Add-Type -AssemblyName System.Speech; '
                 f'$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; '
                 f'$s.Rate = 2; '
                 f'$s.Speak("{letter}")'],
                check=True,
                timeout=5
            )
        except Exception as e:
            print(f"TTS error: {e}")
        finally:
            speech_queue.task_done()

tts_thread = threading.Thread(target=tts_worker, daemon=True)
tts_thread.start()

def speak_prediction(letter):
    global last_speak_time
    current_time = time.time()

    if current_time - last_speak_time >= SPEAK_INTERVAL:
        while not speech_queue.empty():
            try:
                speech_queue.get_nowait()
                speech_queue.task_done()
            except queue.Empty:
                break
        speech_queue.put(letter)
        last_speak_time = current_time

# ============================================================================
# MODEL DEFINITIONS
# ============================================================================

class ResNet50Teacher(nn.Module):
    def __init__(self, num_classes=26):
        super().__init__()
        self.backbone = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V2
        )
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


class MobileNetV2Student(nn.Module):
    def __init__(self, num_classes=26):
        super().__init__()
        self.backbone = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.IMAGENET1K_V2
        )
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.backbone.last_channel, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

# ============================================================================
# INITIALIZATION
# ============================================================================

def setup_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    return device


def load_models(device):
    print("\n" + "="*60)
    print("Loading models...")
    print("="*60)

    teacher_model = ResNet50Teacher(num_classes=NUM_CLASSES).to(device)
    student_model = MobileNetV2Student(num_classes=NUM_CLASSES).to(device)

    if not os.path.exists(TEACHER_MODEL_PATH):
        print(f"❌ ERROR: Teacher model not found at {TEACHER_MODEL_PATH}")
        return None, None

    try:
        checkpoint = torch.load(TEACHER_MODEL_PATH, map_location=device)
        teacher_model.load_state_dict(checkpoint)
        teacher_model.eval()
        print(f"✓ Teacher model loaded: {TEACHER_MODEL_PATH}")
    except Exception as e:
        print(f"❌ Error loading teacher model: {e}")
        return None, None

    if not os.path.exists(STUDENT_MODEL_PATH):
        print(f"❌ ERROR: Student model not found at {STUDENT_MODEL_PATH}")
        return None, None

    try:
        checkpoint = torch.load(STUDENT_MODEL_PATH, map_location=device)
        student_model.load_state_dict(checkpoint)
        student_model.eval()
        print(f"✓ Student model loaded: {STUDENT_MODEL_PATH}")
    except Exception as e:
        print(f"❌ Error loading student model: {e}")
        return None, None

    return teacher_model, student_model


def initialize_mediapipe():
    print("\nInitializing MediaPipe Hands...")
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )

    print("✓ MediaPipe initialized")
    return hands, mp_hands, mp_drawing, mp_drawing_styles


def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

# ============================================================================
# CONSOLE OUTPUT FOR DETECTED LETTERS
# ============================================================================

def log_detection(frame_num, hand_label, pred_results):
    """Print detected letter and confidences to console"""
    if pred_results['valid']:
        print(f"[{frame_num:05d}] {hand_label} hand -> "
              f"Letter: {pred_results['teacher_label']} | "
              f"T:{pred_results['teacher_confidence']:.2f} "
              f"S:{pred_results['student_confidence']:.2f} "
              f"Avg:{pred_results['avg_confidence']:.2f}")
    else:
        print(f"[{frame_num:05d}] {hand_label} hand -> LOW CONFIDENCE "
              f"(T:{pred_results['teacher_confidence']:.2f}, "
              f"S:{pred_results['student_confidence']:.2f})")

# ============================================================================
# HAND PROCESSING
# ============================================================================

def extract_hand_region(frame, hand_landmarks, margin=30):
    h, w, _ = frame.shape

    x_coords = [lm.x * w for lm in hand_landmarks.landmark]
    y_coords = [lm.y * h for lm in hand_landmarks.landmark]

    x_min = max(0, int(min(x_coords)) - margin)
    y_min = max(0, int(min(y_coords)) - margin)
    x_max = min(w, int(max(x_coords)) + margin)
    y_max = min(h, int(max(y_coords)) + margin)

    if x_max - x_min < 50 or y_max - y_min < 50:
        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2
        x_min = max(0, center_x - 50)
        y_min = max(0, center_y - 50)
        x_max = min(w, center_x + 50)
        y_max = min(h, center_y + 50)

    cropped_hand = frame[y_min:y_max, x_min:x_max]
    return cropped_hand, (x_min, y_min, x_max, y_max)


def preprocess_image(cropped_hand, transform, device):
    cropped_rgb = cv2.cvtColor(cropped_hand, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(cropped_rgb)
    img_tensor = transform(pil_image).unsqueeze(0).to(device)
    return img_tensor

# ============================================================================
# INFERENCE
# ============================================================================

def predict_gesture(img_tensor, teacher_model, student_model, device):
    with torch.no_grad():
        teacher_output = teacher_model(img_tensor)
        teacher_probs = F.softmax(teacher_output, dim=1)
        teacher_conf, teacher_idx = torch.max(teacher_probs, 1)
        teacher_label = CLASSES[teacher_idx.item()]
        teacher_confidence = teacher_conf.item()

        student_output = student_model(img_tensor)
        student_probs = F.softmax(student_output, dim=1)
        student_conf, student_idx = torch.max(student_probs, 1)
        student_label = CLASSES[student_idx.item()]
        student_confidence = student_conf.item()

    return {
        'teacher_label': teacher_label,
        'teacher_confidence': teacher_confidence,
        'student_label': student_label,
        'student_confidence': student_confidence,
        'avg_confidence': (teacher_confidence + student_confidence) / 2,
        'valid': teacher_confidence >= CONFIDENCE_THRESHOLD
    }

# ============================================================================
# VISUALIZATION
# ============================================================================

def draw_predictions(frame, bbox, hand_label, pred_results):
    x_min, y_min, x_max, y_max = bbox

    color = (0, 255, 0) if pred_results['valid'] else (0, 0, 255)
    thickness = 3 if pred_results['valid'] else 2
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, thickness)

    cv2.rectangle(frame, (x_min, y_min - 70), (x_min + 350, y_min), (0, 0, 0), -1)
    cv2.rectangle(frame, (x_min, y_max), (x_min + 380, y_max + 70), (0, 0, 0), -1)

    cv2.putText(frame, f"{hand_label} Hand",
                (x_min + 10, y_min - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.putText(frame,
                f"Teacher: {pred_results['teacher_label']} ({pred_results['teacher_confidence']:.2f})",
                (x_min + 10, y_min - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 100), 2)

    cv2.putText(frame,
                f"Student: {pred_results['student_label']} ({pred_results['student_confidence']:.2f})",
                (x_min + 10, y_max + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2)

    ensemble_label = (pred_results['teacher_label']
                      if pred_results['teacher_confidence'] > pred_results['student_confidence']
                      else pred_results['student_label'])

    cv2.putText(frame,
                f"Final: {ensemble_label} ({pred_results['avg_confidence']:.2f})",
                (x_min + 10, y_max + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 2)

# ============================================================================
# MAIN INFERENCE LOOP
# ============================================================================

def run_webcam_inference():
    device = setup_device()
    teacher_model, student_model = load_models(device)

    if teacher_model is None or student_model is None:
        print("\n❌ Failed to load models. Exiting.")
        return

    hands, mp_hands, mp_drawing, mp_drawing_styles = initialize_mediapipe()
    transform = get_transform()

    print("\nInitializing webcam...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ ERROR: Could not open webcam")
        return

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"✓ Webcam opened — Resolution: {frame_width}x{frame_height}")
    print(f"\n✅ Ready! Press 'q' to quit | 's' to save frame")
    print("="*60 + "\n")

    prev_time = 0
    fps_list = []
    frame_num = 0

    with hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame")
                break

            frame_num += 1
            frame = cv2.flip(frame, 1)
            h, w, c = frame.shape

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False
            results = hands.process(frame_rgb)
            frame_rgb.flags.writeable = True

            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_landmarks, handedness in zip(
                    results.multi_hand_landmarks,
                    results.multi_handedness
                ):
                    hand_crop, bbox = extract_hand_region(frame, hand_landmarks, margin=30)

                    if hand_crop.size > 0 and hand_crop.shape[0] > 10 and hand_crop.shape[1] > 10:
                        try:
                            img_tensor = preprocess_image(hand_crop, transform, device)
                            pred_results = predict_gesture(img_tensor, teacher_model, student_model, device)
                            hand_label = handedness.classification[0].label

                            draw_predictions(frame, bbox, hand_label, pred_results)
                            log_detection(frame_num, hand_label, pred_results)

                            ensemble_label = (
                                pred_results['teacher_label']
                                if pred_results['teacher_confidence'] > pred_results['student_confidence']
                                else pred_results['student_label']
                            )
                            speak_prediction(ensemble_label)

                            mp_drawing.draw_landmarks(
                                frame,
                                hand_landmarks,
                                mp_hands.HAND_CONNECTIONS,
                                mp_drawing_styles.get_default_hand_landmarks_style(),
                                mp_drawing_styles.get_default_hand_connections_style()
                            )
                        except Exception as e:
                            print(f"Error processing hand: {e}")
                            continue

            current_time = time.time()
            if current_time - prev_time > 0:
                fps_val = 1 / (current_time - prev_time)
                fps_list.append(fps_val)
                if len(fps_list) > 30:
                    fps_list.pop(0)
                avg_fps = sum(fps_list) / len(fps_list)
                cv2.putText(frame, f"FPS: {avg_fps:.1f}",
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
                prev_time = current_time

            cv2.putText(frame, "Press 'q' to quit | 's' to save frame",
                        (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            cv2.imshow('ASL Recognition - MediaPipe Real-time', frame)

            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('s'):
                filename = f"asl_frame_{frame_num}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Frame saved: {filename}")

    cap.release()
    cv2.destroyAllWindows()
    speech_queue.put(None)  # stop TTS thread cleanly

    if fps_list:
        print(f"\nAverage FPS: {sum(fps_list) / len(fps_list):.2f}")
    print("Inference complete!")

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("ASL Hand Gesture Recognition - Real-time Webcam")
    print("MediaPipe + PyTorch Integration")
    print("="*60)

    if not os.path.exists(TEACHER_MODEL_PATH):
        print(f"\n❌ ERROR: Teacher model not found at {TEACHER_MODEL_PATH}")
        exit(1)

    if not os.path.exists(STUDENT_MODEL_PATH):
        print(f"\n❌ ERROR: Student model not found at {STUDENT_MODEL_PATH}")
        exit(1)

    run_webcam_inference()