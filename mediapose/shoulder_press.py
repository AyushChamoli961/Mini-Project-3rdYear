import mediapipe as mp
import cv2
import numpy as np


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


def count_shoulder_presses():

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)
    card_width, card_height = 180, 100
    card_x, card_y = 10, 10
    card_color = (255, 153, 13)

    with mp_pose.Pose(min_detection_confidence=0.3, min_tracking_confidence=0.7) as pose:
        press_count = 0
        is_pressing = False
        prev_angle_left = None
        prev_angle_right = None
        threshold_angle = 160
        total_reps = 0
        correct_reps = 0
        feedback_text = ""

        while True:
            ret, image = cap.read()

            image = cv2.flip(image, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            results = pose.process(image_rgb)
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            if results.pose_landmarks is not None:
                # Left arm
                left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
                left_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
                left_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]

                # Right arm
                right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                right_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
                right_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]

                angle_left = calculate_angle(
                    left_shoulder, left_elbow, left_wrist)
                angle_right = calculate_angle(
                    right_shoulder, right_elbow, right_wrist)

                print(f'Left angle: {angle_left:.2f} | Right angle: {angle_right:.2f}')

                if prev_angle_left is not None and prev_angle_right is not None:
                    if not is_pressing:
                        if angle_left > 130 and angle_right > 130:
                            is_pressing = True
                            total_reps += 1
                    else:
                        if angle_left < 30 and angle_right < 30:
                            press_count += 1
                            is_pressing = False
                            if total_reps >= 0:
                                correct_reps += 1
                        elif angle_left > 140 or angle_right > 140:
                            feedback_text = "Arms not fully extended"
                        elif angle_left < 15 or angle_right < 15:
                            feedback_text = "Overextension of arms"
                        else:
                            feedback_text = ""

                prev_angle_left = angle_left
                prev_angle_right = angle_right

            cv2.rectangle(image, (10, 10), (360, 110), (0, 0, 0), -1)
            cv2.putText(image, f'Count: {press_count}', (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            if total_reps > 0:
                accuracy = (correct_reps / total_reps) * 100
                cv2.putText(image, f'Accuracy: {accuracy:.2f}%', (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            if feedback_text:
                cv2.putText(image, feedback_text, (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow('Shoulder Press counter', image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    return press_count


if __name__ == "__main__":
    count = count_shoulder_presses()
    print(f"Total shoulder presses counted: {count}")
