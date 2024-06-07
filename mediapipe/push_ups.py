import mediapipe as mp
import cv2
import numpy as np
import argparse


def calculate_angle(a, b, c):
    """
    Calculate the angle between three points using the cosine rule
    """
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])

    radians = np.arccos(np.dot(b - a, c - b) /
                        (np.linalg.norm(b - a) * np.linalg.norm(c - b)))
    angle = np.degrees(radians)

    return angle


def count_pushups(video_source):
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    # Create a video capture object based on the video source
    if isinstance(video_source, int):
        cap = cv2.VideoCapture(video_source)
    else:
        cap = cv2.VideoCapture(video_source)

    card_width, card_height = 180, 100
    card_x, card_y = 10, 10
    card_color = (255, 153, 13)

    with mp_pose.Pose(min_detection_confidence=0.3, min_tracking_confidence=0.7) as pose:
        pushup_count = 0
        is_pushing = False
        prev_angle_right = None
        threshold_angle_start = 160  # Angle to start a pushup
        threshold_angle_end = 80  # Angle to end a pushup
        total_reps = 0
        correct_reps = 0
        feedback_text = ""

        while True:
            ret, image = cap.read()

            if not ret:
                break  # Break the loop if the video has ended

            image = cv2.flip(image, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            results = pose.process(image_rgb)
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            if results.pose_landmarks is not None:
                # Right arm
                right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                right_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
                right_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]

                angle_right = calculate_angle(
                    right_shoulder, right_elbow, right_wrist)

                print(f'Right angle: {angle_right:.2f}')

                if prev_angle_right is not None:
                    if not is_pushing:
                        if angle_right > threshold_angle_start:
                            is_pushing = True
                            total_reps += 1
                    else:
                        if angle_right < threshold_angle_end:
                            pushup_count += 1
                            is_pushing = False
                            if total_reps >= 1:
                                correct_reps += 1
                        elif angle_right > 170:
                            feedback_text = "Overextension of arms"
                        else:
                            feedback_text = ""

                prev_angle_right = angle_right

            cv2.rectangle(image, (10, 10), (360, 110), (0, 0, 0), -1)
            cv2.putText(image, f'Count: {pushup_count}', (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            if total_reps > 0:
                accuracy = (correct_reps / total_reps) * 100
                cv2.putText(image, f'Accuracy: {accuracy:.2f}%', (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            if feedback_text:
                cv2.putText(image, feedback_text, (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Display the frame
            cv2.imshow('Pushup counter', image)

            # Wait for a key press (1 millisecond) and check if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    return pushup_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default=None,
                        help="Path to the video file (optional)")
    args = parser.parse_args()

    if args.video:
        video_source = args.video
    else:
        video_source = 0  # Default to the first available camera

    count = count_pushups(video_source)
    print(f"Total pushups counted: {count}")
