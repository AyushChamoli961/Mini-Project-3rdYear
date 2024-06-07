import mediapipe as mp
import cv2
import numpy as np


def calculate_angle(a, b, c):
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])

    radians = np.arccos(np.dot(b - a, c - b) /
                        (np.linalg.norm(b - a) * np.linalg.norm(c - b)))
    angle = np.degrees(radians)

    return angle


def count_lateral_raises():
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)

    # Cards constraints
    card_width, card_height = 150, 80
    card_x, card_y = 10, 10
    card_color = (255, 153, 13)

    # Accuracy and feedback settings
    correct_reps = 0
    total_reps = 0
    feedback_message = ""
    feedback_color = (0, 0, 255)  # Red for incorrect

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.7) as pose:
        raise_count = 0
        is_raising = False
        prev_angle = None
        prev_angle1 = None
        delta = None
        delta1 = None

        while True:
            ret, image = cap.read()

            image = cv2.flip(image, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            if results.pose_landmarks is not None:
                left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
                left_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
                left_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
                right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                right_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
                right_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]

                angle = calculate_angle(left_shoulder, left_elbow, left_hip)
                angle1 = calculate_angle(
                    right_shoulder, right_elbow, right_hip)

                if prev_angle is not None:
                    delta = abs(angle - prev_angle)
                    delta1 = abs(angle1 - prev_angle1)
                    if not is_raising and delta > 15 and delta1 > 15:
                        is_raising = True
                    elif is_raising and delta < 7 and delta1 < 7:
                        is_raising = False
                        total_reps += 1
                        if angle1 > 120 and angle > 120:
                            raise_count += 1
                            correct_reps += 1
                            feedback_message = "Good rep!"
                            feedback_color = (0, 255, 0)  # Green for correct
                        else:
                            feedback_message = "Raise your arms higher!"
                            feedback_color = (0, 0, 255)  # Red for incorrect

                prev_angle = angle
                prev_angle1 = angle1

            accuracy = (correct_reps / total_reps) * \
                100 if total_reps > 0 else 0

            # Draw feedback card
            cv2.rectangle(image, (card_x, card_y), (card_x +
                          card_width, card_y + card_height), card_color, -1)
            text_x, text_y = card_x + 10, card_y + 30
            text_line1 = 'Lateral Raise'
            text_line2 = f'Count: {raise_count}'
            line_spacing = 30
            font_scale = 0.7
            font_color = (255, 255, 255)

            cv2.putText(image, text_line1, (text_x, text_y),
                        cv2.FONT_HERSHEY_COMPLEX, font_scale, font_color, 2, cv2.LINE_AA)
            cv2.putText(image, text_line2, (text_x, text_y + line_spacing),
                        cv2.FONT_HERSHEY_COMPLEX, font_scale, font_color, 2, cv2.LINE_AA)

            # Draw feedback message
            feedback_x, feedback_y = 10, card_y + card_height + 40
            cv2.putText(image, feedback_message, (feedback_x, feedback_y),
                        cv2.FONT_HERSHEY_COMPLEX, font_scale, feedback_color, 2, cv2.LINE_AA)

            # Draw accuracy
            accuracy_text = f'Accuracy: {accuracy:.2f}%'
            cv2.putText(image, accuracy_text, (feedback_x, feedback_y + line_spacing),
                        cv2.FONT_HERSHEY_COMPLEX, font_scale, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow('Lateral Raise counter', image)

            if cv2.waitKey(1) == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    return raise_count


if __name__ == "__main__":
    count = count_lateral_raises()
    print(count)



# python3 shoulder_press.py - -video "../demo-videos/shoulder_press.mp4"