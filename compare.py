import mediapipe as mp
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from misc.utils import find_person_id_associations
from misc.visualization import draw_points_and_skeleton, joints_dict
from model import SimpleHRNet
import os
import sys
import argparse
import ast
import cv2
import time
import torch
from vidgear.gears import CamGear
import numpy as np

import matplotlib
matplotlib.use('Agg')

keypoints = {
    0: "nose",
    1: "left_eye",
    2: "right_eye",
    3: "left_ear",
    4: "right_ear",
    5: "left_shoulder",
    6: "right_shoulder",
    7: "left_elbow",
    8: "right_elbow",
    9: "left_wrist",
    10: "right_wrist",
    11: "left_hip",
    12: "right_hip",
    13: "left_knee",
    14: "right_knee",
    15: "left_ankle",
    16: "right_ankle"
}

mideiapose_angle_data = []
hrnet_angle_data = []


def save_graph(angle_data, additional_data):
    plt.figure(figsize=(10, 6))
    plt.plot(angle_data, label='Angle Data')
    plt.plot(additional_data, label='Additional Data')
    plt.title('Angle Over Time')
    plt.xlabel('Frame')
    plt.ylabel('Angle (degrees)')
    plt.legend()
    plt.savefig('angle_plot.png', dpi=300, bbox_inches='tight')


def update_graph(angle_data):
    angle_plot.clear()
    angle_plot.plot(angle_data)
    angle_plot.set_xlabel('Frame')
    angle_plot.set_ylabel('Angle (degrees)')
    angle_plot.set_title('Angle Over Time')
    canvas.draw()
    root.update()


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


def count_shoulder_presses(video_source):

    angle_data = []
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
        press_count = 0
        is_pressing = False
        prev_angle_left = None
        prev_angle_right = None
        threshold_angle_start = 130  # Angle to start a rep
        threshold_angle_end = 30  # Angle to end a rep
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

                angle_avg = (angle_left + angle_right) / 2
                mideiapose_angle_data.append(angle_avg)

                print(
                    f'Left angle: {angle_left:.2f} | Right angle: {angle_right:.2f}')

                if prev_angle_left is not None and prev_angle_right is not None:
                    if not is_pressing:
                        if angle_left > threshold_angle_start and angle_right > threshold_angle_start:
                            is_pressing = True
                            total_reps += 1
                    else:
                        if angle_left < threshold_angle_end and angle_right < threshold_angle_end:
                            press_count += 1
                            is_pressing = False
                            if total_reps >= 1:
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

            # Display the frame
            cv2.imshow('Shoulder Press counter', image)

            # Wait for a key press (1 millisecond) and check if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                save_graph(angle_data)
                break

    cap.release()
    cv2.destroyAllWindows()

    return press_count


def main(camera_id, filename, hrnet_m, hrnet_c, hrnet_j, hrnet_weights, hrnet_joints_set, image_resolution, single_person, use_tiny_yolo, disable_tracking, max_batch_size, disable_vidgear, save_video, video_format, video_framerate, device, exercise_type):
    if device is not None:
        device = torch.device(device)
    else:
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

    image_resolution = ast.literal_eval(image_resolution)
    has_display = 'DISPLAY' in os.environ.keys() or sys.platform == 'win32'
    video_writer = None

    if filename is not None:
        video = cv2.VideoCapture(filename)
        assert video.isOpened()
    elif disable_vidgear:
        video = cv2.VideoCapture(camera_id)
        assert video.isOpened()
    else:
        video = CamGear(camera_id).start()

    # Rest of the code...

    if use_tiny_yolo:
        yolo_model_def = "./models/detectors/yolo/config/yolov3-tiny.cfg"
        yolo_class_path = "./models/detectors/yolo/data/coco.names"
        yolo_weights_path = "./models/detectors/yolo/weights/yolov3-tiny.weights"
    else:
        yolo_model_def = "./models/detectors/yolo/config/yolov3.cfg"
        yolo_class_path = "./models/detectors/yolo/data/coco.names"
        yolo_weights_path = "./models/detectors/yolo/weights/yolov3.weights"

    model = SimpleHRNet(
        hrnet_c,
        hrnet_j,
        hrnet_weights,
        model_name=hrnet_m,
        resolution=image_resolution,
        multiperson=not single_person,
        return_heatmaps=False,
        return_bounding_boxes=not disable_tracking,
        max_batch_size=max_batch_size,
        yolo_model_def=yolo_model_def,
        yolo_class_path=yolo_class_path,
        yolo_weights_path=yolo_weights_path,
        device=device
    )

    if not disable_tracking:
        prev_boxes = None
        prev_pts = None
        prev_person_ids = None
        next_person_id = 0

    flag = 0
    prev_flag = flag
    counter = 0
    data = 0
    angle = 0
    prev_data = data

    angle_data = []
    pts_data = []

    while True:
        t = time.time()
        if filename is not None:
            ret, frame = video.read()
        else:
            frame = video.read()
        if frame is None:
            break

        pts = model.predict(frame)
        pts_data.append(pts)

        if not disable_tracking:
            boxes, pts = pts
            if len(pts) > 0:
                if prev_pts is None and prev_person_ids is None:
                    person_ids = np.arange(next_person_id, len(
                        pts) + next_person_id, dtype=np.int32)
                    next_person_id = len(pts) + 1
                else:
                    boxes, pts, person_ids = find_person_id_associations(
                        boxes=boxes, pts=pts, prev_boxes=prev_boxes, prev_pts=prev_pts, prev_person_ids=prev_person_ids,
                        next_person_id=next_person_id, pose_alpha=0.2, similarity_threshold=0.4, smoothing_alpha=0.1,
                    )
                    next_person_id = max(
                        next_person_id, np.max(person_ids) + 1)

            else:
                person_ids = np.array((), dtype=np.int32)

            prev_boxes = boxes.copy()
            prev_pts = pts.copy()
            prev_person_ids = person_ids
        else:
            person_ids = np.arange(len(pts), dtype=np.int32)

        for i, (pt, pid) in enumerate(zip(pts, person_ids)):
            frame, data, angle = draw_points_and_skeleton(frame, pt, joints_dict(
            )[hrnet_joints_set]['skeleton'], person_index=pid, exercise_type=exercise_type)

        frame = cv2.rectangle(
            frame, (0, 0), (int(frame.shape[1]*0.7), int(frame.shape[0]*0.1)), (0, 0, 0), -1)

        fps = 1. / (time.time() - t)
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (int(frame.shape[1]*0.01), int(frame.shape[0]*0.035))
        fontScale = frame.shape[0] * 0.0014
        color = (255, 255, 255)
        thickness = 1
        frame = cv2.putText(frame, 'FPS: {:.3f}'.format(fps), org, font,
                            fontScale*0.35, color, thickness, cv2.LINE_AA)

        if exercise_type == 1:  # for pushUps
            angle_data.append(data)
            print(data)
            if (len(pts) > 0):

                if (data > 165):
                    flag = 0
                if (data < 70):
                    flag = 1
                if (prev_flag == 1 and flag == 0):
                    counter = counter+1

            prev_flag = flag

            org = (int(frame.shape[1]*0.01), int(frame.shape[0]*0.08))
            text = "PushUps Count="+str(counter)
            frame = cv2.putText(frame, text, org, font,
                                fontScale, color, thickness*2, cv2.LINE_AA)

        elif exercise_type == 2:  # for Squats
            print(data)
            angle_data.append(data)
            if (len(pts) > 0):
                if (data > 150):
                    flag = 0
                if (data < 90):
                    flag = 1
                if (prev_flag == 1 and flag == 0):
                    counter = counter+1

            prev_flag = flag

            org = (int(frame.shape[1]*0.01), int(frame.shape[0]*0.08))
            text = "Squat Count="+str(counter)
            frame = cv2.putText(frame, text, org, font,
                                fontScale, color, thickness*2, cv2.LINE_AA)

        elif exercise_type == 3:  # for Shoulder Press
            print(angle)
            hrnet_angle_data.append(angle)
            if len(pts) > 0:
                if angle > 140:
                    flag = 0  # Arm bent
                elif angle < 70:
                    flag = 1  # Arm extended

                # Check if the previous rep was incomplete
                if prev_flag == 0 and flag == 2:
                    incomplete_rep_text = " rep is incomplete!"
                    cv2.putText(frame, incomplete_rep_text, (10, 60), cv2.FONT_HERSHEY_TRIPLEX,
                                fontScale * 2, (0, 0, 255), thickness * 2, cv2.LINE_AA)
                    # White background
                    cv2.rectangle(frame, (5, 50),
                                  (frame.shape[1] // 4, 80), (255, 255, 255), -1)

                # Check if the user is overextending the arms
                if angle > 180:
                    overextension_text = "Overextending arms!"
                    cv2.putText(frame, overextension_text, (10, 90), cv2.FONT_HERSHEY_TRIPLEX,
                                fontScale * 2, (0, 0, 255), thickness * 2, cv2.LINE_AA)
                    # White background
                    cv2.rectangle(frame, (5, 80),
                                  (frame.shape[1] // 4, 100), (255, 255, 255), -1)

                # Count reps
                if prev_flag == 1 and flag == 0:
                    counter += 1

            prev_flag = flag

            org = (int(frame.shape[1] * 0.01), int(frame.shape[0] * 0.08))
            text = "Shoulder Press Count=" + str(counter)
            frame = cv2.putText(frame, text, org, font, fontScale,
                                color, thickness * 2, cv2.LINE_AA)
        elif exercise_type == 4:  # for dumbell curl
            print(data)
            angle_data.append(data)
            if (len(pts) > 0):
                if (data > 150):
                    flag = 0
                if (data < 60):
                    flag = 1
                if (prev_flag == 1 and flag == 0):
                    counter = counter+1

            prev_flag = flag

            org = (int(frame.shape[1]*0.01), int(frame.shape[0]*0.08))
            text = "Dumbell Curl Count="+str(counter)
            frame = cv2.putText(frame, text, org, font,
                                fontScale, color, thickness*2, cv2.LINE_AA)

        elif exercise_type == 5:  # for dumbell side lateral
            print(angle)
            angle_data.append(angle)
            if (len(pts) > 0):
                if (angle > 120):
                    flag = 0
                if (angle < 30):
                    flag = 1
                if (prev_flag == 1 and flag == 0):
                    counter = counter+1

            prev_flag = flag

            org = (int(frame.shape[1]*0.01), int(frame.shape[0]*0.08))
            text = "Dumbell Side Count="+str(counter)
            frame = cv2.putText(frame, text, org, font,
                                fontScale, color, thickness*2, cv2.LINE_AA)

    ########################################################################################################

        cv2.imshow('frame.png', frame)
        k = cv2.waitKey(1)
        if k == 27:  # Esc button
            print("Total Reps: ", counter, angle)
            save_graph(hrnet_angle_data)
            if disable_vidgear:
                video.release()
            else:
                video.stop()
            break

        if save_video:
            if video_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*video_format)  # video format
                video_writer = cv2.VideoWriter(
                    'arnleft.avi', fourcc, video_framerate, (frame.shape[1], frame.shape[0]))
            video_writer.write(frame)

    if save_video:
        video_writer.release()
    save_graph(hrnet_angle_data, mideiapose_angle_data)
    print(pts_data)
    plot_and_save_joint_positions(pts_data, keypoints)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--camera_id", "-d", help="open the camera with the specified id", type=int, default=0)
    parser.add_argument(
        "--filename", "-f", help="open the specified video (overrides the --camera_id option)", type=str, default=None)

    # type=str, default='squats.mp4')
    parser.add_argument("--exercise_type", "-et",
                        help="1 for pushups, 2 for squats, 3 for pullups 4 for dumbell curl 5 for dumbell side curl", type=int, required=True)
    parser.add_argument("--hrnet_weights", "-w", help="hrnet parameters - path to the pretrained weights",
                        type=str, default="./weights/w32_256x192.pth")
    parser.add_argument("--image_resolution", "-r",
                        help="image resolution", type=str, default='(256,192)')
    # help="image resolution", type=str, default='(384, 288)')
    # parser.add_argument("--filename", "-f", help="open the specified video (overrides the --camera_id option)",
    #                     type=str, default=None)
    parser.add_argument(
        "--hrnet_j", "-j", help="hrnet parameters - number of joints", type=int, default=17)
    parser.add_argument(
        "--hrnet_m", "-m", help="network model - 'HRNet' or 'PoseResNet'", type=str, default='HRNet')
    parser.add_argument("--hrnet_c", "-c", help="hrnet parameters - number of channels (if model is HRNet), "
                        "resnet size (if model is PoseResNet)", type=int, default=32)
    parser.add_argument("--hrnet_joints_set",
                        help="use the specified set of joints ('coco' and 'mpii' are currently supported)",
                        type=str, default="coco")
    parser.add_argument("--single_person",
                        help="disable the multiperson detection (YOLOv3 or an equivalen detector is required for"
                        "multiperson detection)",
                        action="store_true", default=True)
    parser.add_argument("--use_tiny_yolo",
                        help="Use YOLOv3-tiny in place of YOLOv3 (faster person detection). Ignored if --single_person",
                        action="store_true")
    parser.add_argument("--disable_tracking",
                        help="disable the skeleton tracking and temporal smoothing functionality",
                        action="store_true")
    parser.add_argument(
        "--max_batch_size", help="maximum batch size used for inference", type=int, default=16)
    parser.add_argument("--disable_vidgear",
                        help="disable vidgear (which is used for slightly better realtime performance)",
                        action="store_true")  # see https://pypi.org/project/vidgear/
    parser.add_argument(
        "--save_video", help="save output frames into a video.", action="store_false")
    parser.add_argument("--video_format", help="fourcc video format. Common formats: `MJPG`, `XVID`, `X264`."
                        "See http://www.fourcc.org/codecs.php", type=str, default='MJPG')
    parser.add_argument("--video_framerate",
                        help="video framerate", type=float, default=30)
    parser.add_argument("--device", help="device to be used (default: cuda, if available)."
                        "Set to `cuda` to use all available GPUs (default); "
                        "set to `cuda:IDS` to use one or more specific GPUs "
                                         "(e.g. `cuda:0` `cuda:1,2`); "
                                         "set to `cpu` to run on cpu.", type=str, default=None)
    args = parser.parse_args()
    if args.filename:
        video_source = args.filename
    else:
        video_source = 0  # Default to the first available camera
    count = count_shoulder_presses(video_source)
    print(f"Total shoulder presses counted: {count}")
    main(**args.__dict__)


# python main.py --filename 'demo_videos/squat.mp4' --exercise_type 2
