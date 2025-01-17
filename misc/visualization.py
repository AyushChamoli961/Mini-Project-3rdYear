import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import ffmpeg
import math
# points_color_palette='gist_rainbow', skeleton_color_palette='jet',
# points_palette_samples=10,

colors = np.round(np.array(plt.get_cmap('gist_rainbow')(
    np.linspace(0, 1, 16))) * 255).astype(np.uint8)[:, -2::-1].tolist()




def joints_dict():
    joints = {
        "coco": {
            "keypoints": {
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
            },
            "skeleton": [
                # # [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8],
                # # [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]
                # [15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6], [5, 7],
                # [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6]
                [15, 13], [13, 11], [16, 14], [14, 12], [
                    11, 12], [5, 11], [6, 12], [5, 6], [5, 7],
                [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [
                    0, 2], [1, 3], [2, 4],  # [3, 5], [4, 6]
                [0, 5], [0, 6]
            ]
        },
        "mpii": {
            "keypoints": {
                0: "right_ankle",
                1: "right_knee",
                2: "right_hip",
                3: "left_hip",
                4: "left_knee",
                5: "left_ankle",
                6: "pelvis",
                7: "thorax",
                8: "upper_neck",
                9: "head top",
                10: "right_wrist",
                11: "right_elbow",
                12: "right_shoulder",
                13: "left_shoulder",
                14: "left_elbow",
                15: "left_wrist"
            },
            "skeleton": [
                # [5, 4], [4, 3], [0, 1], [1, 2], [3, 2], [13, 3], [12, 2], [13, 12], [13, 14],
                # [12, 11], [14, 15], [11, 10], # [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]
                [5, 4], [4, 3], [0, 1], [1, 2], [3, 2], [
                    3, 6], [2, 6], [6, 7], [7, 8], [8, 9],
                [13, 7], [12, 7], [13, 14], [12, 11], [14, 15], [11, 10],
            ]
        },
    }
    return joints


def draw_points_front(image, points, exercise_type, confidence_threshold=0.5):
    circle_size = max(1, min(image.shape[:2]) // 160)

    y0 = 0
    y1 = 0
    y2 = 0
    ylw = points[9][0]
    yrw = points[10][0]
    yls = points[5][0]
    yrs = points[6][0]
    z0, z1, z2 = 0, 0, 0
    elbow_angle = 0  # Initialize elbow angle

    for i, pt in enumerate(points):
        if pt[2] > confidence_threshold:
            image = cv2.circle(image, (int(pt[1]), int(
                pt[0])), circle_size, tuple(colors[i % len(colors)]), -1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            org = (int(pt[1]), int(pt[0]))  # Convert coordinates to integers
            fontScale = 1
            color = (255, 255, 255)
            thickness = 2
            image = cv2.putText(image, str(i), org, font,
                                fontScale, color, thickness, cv2.LINE_AA)

        if exercise_type == 3:
            if i == 0:
                y0 = pt[0]
                z0 = pt[2]
            if i == 1:
                y1 = pt[0]
                z1 = pt[2]
            if i == 2:
                y2 = pt[0]
                z2 = pt[2]
            dist = distance(y0, y1, y2, z0, z1, z2, ylw, yrw)

            # Calculate elbow angle for exercise type 3
            if i == 5:  # Left shoulder
                x1, y1 = pt[1], pt[0]
            elif i == 7:  # Left elbow
                x2, y2 = pt[1], pt[0]
            elif i == 9:  # Left wrist
                x3, y3 = pt[1], pt[0]
                elbow_angle = angle(x1, y1, x2, y2, x3, y3)

        if exercise_type == 5:
            dist = distance_dumbell(yls, yrs, ylw, yrw)
            if i == 6:  # Left shoulder
                x1, y1 = pt[1], pt[0]
            elif i == 10:  # Left elbow
                x2, y2 = pt[1], pt[0]
            elif i == 12:  # Left wrist
                x3, y3 = pt[1], pt[0]
                elbow_angle = angle(x1, y1, x2, y2, x3, y3)

    return image, dist, elbow_angle


def draw_points_one_side(image, points, exercise_type, confidence_threshold=0.5):
    data = 0    
    # ToDo Shape it taking into account the size of the detection
    circle_size = max(1, min(image.shape[:2]) // 160)
    x1 = 0
    y1 = 0
    x2 = 0
    y2 = 0
    x3 = 0
    y3 = 0
    xn = points[0][1]
    xlh = points[11][1]
    xrh = points[12][1]

    if exercise_type == 1 or exercise_type == 4:
        if (xn < xlh or xn < xrh):
            a, b, c = 5, 7, 9
        else:
            a, b, c = 6, 8, 10
    elif exercise_type == 2:
        if (xn < xlh or xn < xrh):
            a, b, c = 11, 13, 15
        else:
            a, b, c = 12, 14, 16

    for i, pt in enumerate(points):
        if pt[2] > confidence_threshold:
            image = cv2.circle(image, (int(pt[1]), int(
                pt[0])), circle_size, tuple(colors[i % len(colors)]), -1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            org = (int(pt[1]), int(pt[0]))  # Convert coordinates to integers
            fontScale = 1
            color = (255, 255, 255)
            thickness = 2
            image = cv2.putText(image, str(i), org, font,
                                fontScale, color, thickness, cv2.LINE_AA)

        if i == a:
            x1 = pt[1]
            y1 = pt[0]
        if i == b:
            x2 = pt[1]
            y2 = pt[0]
        if i == c:
            x3 = pt[1]
            y3 = pt[0]

        ang = angle(x1, y1, x2, y2, x3, y3)

    return image, ang, data



def draw_skeleton(image, points, skeleton, person_index=0, confidence_threshold=0.5):
    """
    Draws a `skeleton` on `image`.

    Args:
        image: image in opencv format
        points: list of points to be drawn.
            Shape: (nof_points, 3)
            Format: each point should contain (y, x, confidence)
        skeleton: list of joints to be drawn
            Shape: (nof_joints, 2)
            Format: each joint should contain (point_a, point_b) where `point_a` and `point_b` are an index in `points`
        person_index: index of the person in `image`
            Default: 0
        confidence_threshold: only points with a confidence higher than this threshold will be drawn. Range: [0, 1]
            Default: 0.5

    Returns:
        A new image with overlaid joints

    """
    for i, joint in enumerate(skeleton):
        pt1, pt2 = points[joint]
        if pt1[2] > confidence_threshold and pt2[2] > confidence_threshold:
            image = cv2.line(
                image, (int(pt1[1]), int(pt1[0])), (int(pt2[1]), int(pt2[0])),
                tuple(colors[i % len(colors)]), 2
                # tuple(colors[person_index % len(colors)]), 2
            )

    return image


def draw_points_and_skeleton(image, points, skeleton, person_index=0,
                             confidence_threshold=0.5, exercise_type=1):

    image = draw_skeleton(image, points, skeleton, person_index=person_index,
                          confidence_threshold=confidence_threshold)
    # plt.imshow(image)
    # plt.show()

    if  exercise_type == 1 or exercise_type == 2 or str(exercise_type)[0] == str(4):
        image, data, angle  = draw_points_one_side(
            image, points, exercise_type,  confidence_threshold=confidence_threshold)

    elif exercise_type == 3 or exercise_type == 5:
        image, data, angle = draw_points_front(
            image, points, exercise_type, confidence_threshold=confidence_threshold)
    return image, data, angle


def save_images(images, target, joint_target, output, joint_output, joint_visibility, summary_writer=None, step=0,
                prefix=''):
    """
    Creates a grid of images with gt joints and a grid with predicted joints.
    This is a basic function for debugging purposes only.

    If summary_writer is not None, the grid will be written in that SummaryWriter with name "{prefix}_images" and
    "{prefix}_predictions".

    Args:
        images (torch.Tensor): a tensor of images with shape (batch x channels x height x width).
        target (torch.Tensor): a tensor of gt heatmaps with shape (batch x channels x height x width).
        joint_target (torch.Tensor): a tensor of gt joints with shape (batch x joints x 2).
        output (torch.Tensor): a tensor of predicted heatmaps with shape (batch x channels x height x width).
        joint_output (torch.Tensor): a tensor of predicted joints with shape (batch x joints x 2).
        joint_visibility (torch.Tensor): a tensor of joint visibility with shape (batch x joints).
        summary_writer (tb.SummaryWriter): a SummaryWriter where write the grids.
            Default: None
        step (int): summary_writer step.
            Default: 0
        prefix (str): summary_writer name prefix.
            Default: ""

    Returns:
        A pair of images which are built from torchvision.utils.make_grid
    """
    # Input images with gt
    images_ok = images.detach().clone()
    images_ok[:, 0].mul_(0.229).add_(0.485)
    images_ok[:, 1].mul_(0.224).add_(0.456)
    images_ok[:, 2].mul_(0.225).add_(0.406)
    for i in range(images.shape[0]):
        joints = joint_target[i] * 4.
        joints_vis = joint_visibility[i]

        for joint, joint_vis in zip(joints, joints_vis):
            if joint_vis[0]:
                a = int(joint[1].item())
                b = int(joint[0].item())
                # images_ok[i][:, a-1:a+1, b-1:b+1] = torch.tensor([1, 0, 0])
                images_ok[i][0, a - 1:a + 1, b - 1:b + 1] = 1
                images_ok[i][1:, a - 1:a + 1, b - 1:b + 1] = 0
    grid_gt = torchvision.utils.make_grid(images_ok, nrow=int(
        images_ok.shape[0] ** 0.5), padding=2, normalize=False)
    if summary_writer is not None:
        summary_writer.add_image(prefix + 'images', grid_gt, global_step=step)

    # Input images with prediction
    images_ok = images.detach().clone()
    images_ok[:, 0].mul_(0.229).add_(0.485)
    images_ok[:, 1].mul_(0.224).add_(0.456)
    images_ok[:, 2].mul_(0.225).add_(0.406)
    for i in range(images.shape[0]):
        joints = joint_output[i] * 4.
        joints_vis = joint_visibility[i]

        for joint, joint_vis in zip(joints, joints_vis):
            if joint_vis[0]:
                a = int(joint[1].item())
                b = int(joint[0].item())
                # images_ok[i][:, a-1:a+1, b-1:b+1] = torch.tensor([1, 0, 0])
                images_ok[i][0, a - 1:a + 1, b - 1:b + 1] = 1
                images_ok[i][1:, a - 1:a + 1, b - 1:b + 1] = 0
    grid_pred = torchvision.utils.make_grid(images_ok, nrow=int(
        images_ok.shape[0] ** 0.5), padding=2, normalize=False)
    if summary_writer is not None:
        summary_writer.add_image(
            prefix + 'predictions', grid_pred, global_step=step)

    return grid_gt, grid_pred


def angle(x1, y1, x2, y2, x3, y3):
    a = math.sqrt((x3-x2)**2+(y3-y2)**2)
    b = math.sqrt((x3-x1)**2+(y3-y1)**2)
    c = math.sqrt((x2-x1)**2+(y2-y1)**2)
    if a == 0 or c == 0:
        angle = 0
        return angle
    else:
        term = (a**2+c**2-b**2)/(2*c*a)
        if term > 1:
            term = 1
        elif term < -1:
            term = -1
        angle_rad = math.acos(term)
        angle = (180*angle_rad)/(math.pi)
    return angle


def distance(y0, y1, y2, z0, z1, z2, ylw, yrw):
    t1, t2, t3 = 0, 0, 0
    if(z0 > 0.5 and y0 > ylw and y0 > yrw):
        t1 = 1
    if(z1 > 0.5 and y1 > ylw and y1 > yrw):
        t2 = 1
    if(z2 > 0.5 and y2 > ylw and y2 > yrw):
        t3 = 1

    if(t1 == 1 and t2 == 1):
        return 1
    if(t1 == 1 and t3 == 1):
        return 1
    if(t2 == 1 and t3 == 1):
        return 1

    return -1


def distance_dumbell(yls, yrs, ylw, yrw):
    if (yrs > yrw):
        return 1
    if (yls > ylw):
        return 1
    return -1
