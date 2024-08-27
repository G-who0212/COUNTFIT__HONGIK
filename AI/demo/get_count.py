from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import csv
import sys, os
import shutil

# from PIL import Image
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision
import cv2
import numpy as np
import time
import math
import boto3

sys.path.append(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
# sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# import lib.models
from lib.config import cfg
from lib.config import update_config
from lib.core.function import get_final_preds
from lib.utils.transforms import get_affine_transform
from lib.models.pose_hrnet import get_pose_net

COCO_KEYPOINT_INDEXES = {
    0: 'nose',
    1: 'left_eye',
    2: 'right_eye',
    3: 'left_ear',
    4: 'right_ear',
    5: 'left_shoulder',
    6: 'right_shoulder',
    7: 'left_elbow',
    8: 'right_elbow',
    9: 'left_wrist',
    10: 'right_wrist',
    11: 'left_hip',
    12: 'right_hip',
    13: 'left_knee',
    14: 'right_knee',
    15: 'left_ankle',
    16: 'right_ankle'
}

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

SKELETON = [
    [1,3],[1,0],[2,4],[2,0],[0,5],[0,6],[5,7],[7,9],[6,8],[8,10],[5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16]
]

CocoColors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

NUM_KPTS = 17

PUSH_UP = 0
PULL_UP = 1
SQUAT = 2

CTX = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def draw_pose(keypoints,img):
    """draw the keypoints and the skeletons.
    :params keypoints: the shape should be equal to [17,2]
    :params img:
    """
    assert keypoints.shape == (NUM_KPTS,2)
    for i in range(len(SKELETON)):
        kpt_a, kpt_b = SKELETON[i][0], SKELETON[i][1]
        x_a, y_a = keypoints[kpt_a][0],keypoints[kpt_a][1]
        x_b, y_b = keypoints[kpt_b][0],keypoints[kpt_b][1] 
        cv2.circle(img, (int(x_a), int(y_a)), 6, CocoColors[i], -1)
        cv2.circle(img, (int(x_b), int(y_b)), 6, CocoColors[i], -1)
        cv2.line(img, (int(x_a), int(y_a)), (int(x_b), int(y_b)), CocoColors[i], 2)

def draw_bbox(box,img):
    """draw the detected bounding box on the image.
    :param img:
    """
    cv2.rectangle(img, box[0], box[1], color=(0, 255, 0),thickness=3)


def get_person_detection_boxes(model, img, threshold=0.5):
    pred = model(img)
    pred_classes = [COCO_INSTANCE_CATEGORY_NAMES[i]
                    for i in list(pred[0]['labels'].cpu().numpy())]  # Get the Prediction Score
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])]
                  for i in list(pred[0]['boxes'].detach().cpu().numpy())]  # Bounding boxes
    pred_score = list(pred[0]['scores'].detach().cpu().numpy())
    if not pred_score or max(pred_score)<threshold:
        return []
    # Get list of index with score greater than threshold
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]
    pred_boxes = pred_boxes[:pred_t+1]
    pred_classes = pred_classes[:pred_t+1]

    person_boxes = []
    for idx, box in enumerate(pred_boxes):
        if pred_classes[idx] == 'person':
            person_boxes.append(box)

    return person_boxes


def get_pose_estimation_prediction(pose_model, image, center, scale):
    rotation = 0

    # pose estimation transformation
    trans = get_affine_transform(center, scale, rotation, cfg.MODEL.IMAGE_SIZE)
    model_input = cv2.warpAffine(
        image,
        trans,
        (int(cfg.MODEL.IMAGE_SIZE[0]), int(cfg.MODEL.IMAGE_SIZE[1])),
        flags=cv2.INTER_LINEAR)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # pose estimation inference
    model_input = transform(model_input).unsqueeze(0)
    # switch to evaluate mode
    pose_model.eval()
    with torch.no_grad():
        # compute output heatmap
        output = pose_model(model_input)
        preds, _ = get_final_preds(
            cfg,
            output.clone().cpu().numpy(),
            np.asarray([center]),
            np.asarray([scale]))

        return preds


def box_to_center_scale(box, model_image_width, model_image_height):
    """convert a box to center,scale information required for pose transformation
    Parameters
    ----------
    box : list of tuple
        list of length 2 with two tuples of floats representing
        bottom left and top right corner of a box
    model_image_width : int
    model_image_height : int

    Returns
    -------
    (numpy array, numpy array)
        Two numpy arrays, coordinates for the center of the box and the scale of the box
    """
    center = np.zeros((2), dtype=np.float32)

    bottom_left_corner = box[0]
    top_right_corner = box[1]
    box_width = top_right_corner[0]-bottom_left_corner[0]
    box_height = top_right_corner[1]-bottom_left_corner[1]
    bottom_left_x = bottom_left_corner[0]
    bottom_left_y = bottom_left_corner[1]
    center[0] = bottom_left_x + box_width * 0.5
    center[1] = bottom_left_y + box_height * 0.5

    aspect_ratio = model_image_width * 1.0 / model_image_height
    pixel_std = 200

    if box_width > aspect_ratio * box_height:
        box_height = box_width * 1.0 / aspect_ratio
    elif box_width < aspect_ratio * box_height:
        box_width = box_height * aspect_ratio
    scale = np.array(
        [box_width * 1.0 / pixel_std, box_height * 1.0 / pixel_std],
        dtype=np.float32)
    if center[0] != -1:
        scale = scale * 1.25

    return center, scale

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg', type=str, default='/Users/gwho/Desktop/CountFit/AI/demo/inference-config.yaml')

    parser.add_argument('opts',
                        help='Modify config options using the command-line',
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    # args expected by supporting codebase  
    args.modelDir = ''
    args.logDir = ''
    args.dataDir = ''
    args.prevModelDir = ''
    return args

def get_cosine(v1, v2):
    """
    Args:
        v1: type은 list, 두 벡터 사이의 cosine 값을 얻기 위한 벡터
        v2: type은 list, 두 벡터 사이의 cosine 값을 얻기 위한 벡터
    """
    return (v1[0]*v2[0] + v1[1]*v2[1])/((abs(math.sqrt(v1[0]**2 + v1[1]**2)))*(abs(math.sqrt(v2[0]**2 + v2[1]**2))))

def predict_image(image, chk, work_out_type, box_model, pose_model):

    image_bgr = image
    work_out = work_out_type    
    count = False
    # chk = 0 # count를 셀만한 정도로 움직였는지  check / 0: 준비X, 1: 운동 준비O, 2: count를 셀만큼의 가동범위 동작
    pushup_threshold1 = math.cos(math.pi*2/3) # 120도
    pushup_threshold2 = math.cos(math.pi*7/9) # 140도
    pullup_threshold1 = math.cos(math.pi/2) # 90도
    pullup_threshold2 = math.cos(math.pi*5/9) # 100도
    squat_threshold1 = math.cos(math.pi/2) # 90도
    squat_threshold2 = math.cos(math.pi*2/3) # 120도
    
    pushup_max_len = 0
    
    # estimate on the image
    last_time = time.time()
    image = image_bgr[:, :, [2, 1, 0]]

    input = []
    img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img/255.).permute(2,0,1).float().to(CTX)
    input.append(img_tensor)

    # object detection box
    pred_boxes = get_person_detection_boxes(box_model, input, threshold=0.9)

    # pose estimation
    if len(pred_boxes) >= 1:
        # for box in pred_boxes:
        #     center, scale = box_to_center_scale(box, cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1])
        #     image_pose = image.copy() if cfg.DATASET.COLOR_RGB else image_bgr.copy()
        #     pose_preds = get_pose_estimation_prediction(pose_model, image_pose, center, scale)
        #     if len(pose_preds)>=1:
        #         for kpt in pose_preds:
        #             draw_pose(kpt,image_bgr) # draw the poses
        for box in pred_boxes:
            center, scale = box_to_center_scale(box, cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1])
            image_pose = image.copy() if cfg.DATASET.COLOR_RGB else image_bgr.copy()
            pose_preds = get_pose_estimation_prediction(pose_model, image_pose, center, scale)

            # 중앙에 사람이 있지 않으면 예측하지 않음(다른 사람 중복 예측 방지)
            if (center[0] < image_pose.shape[1]/2.0 - image_pose.shape[1]/6.0) or (center[0] > image_pose.shape[1]/2.0 + image_pose.shape[1]/6.0):
                continue
            
            # 5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow, 9: left_wrist, 10: right_wrist
            # 11: left_hip, 12: right_hip, 13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle
            
            if work_out == PUSH_UP:
                
                # 팔꿈치->어께 벡터, 팔꿈치->손목 벡터 구하기
                left_elbow_to_shoulder = [pose_preds[0][5][0] - pose_preds[0][7][0], pose_preds[0][5][1] - pose_preds[0][7][1]]
                right_elbow_to_shoulder = [pose_preds[0][6][0] - pose_preds[0][8][0], pose_preds[0][6][1] - pose_preds[0][8][1]]
                left_elbow_to_wrist = [pose_preds[0][9][0] - pose_preds[0][7][0], pose_preds[0][9][1] - pose_preds[0][7][1]]
                right_elbow_to_wrist = [pose_preds[0][10][0] - pose_preds[0][8][0], pose_preds[0][10][1] - pose_preds[0][8][1]]
                left_elbow_cos = get_cosine(left_elbow_to_shoulder, left_elbow_to_wrist)
                right_elbow_cos = get_cosine(right_elbow_to_shoulder, right_elbow_to_wrist)
                
                # 엉덩이->발목 벡터 구하기
                left_hip_to_ankle = [pose_preds[0][15][0] - pose_preds[0][11][0], pose_preds[0][15][1] - pose_preds[0][11][1]]
                right_hip_to_ankle = [pose_preds[0][16][0] - pose_preds[0][12][0], pose_preds[0][16][1] - pose_preds[0][12][1]]
                # 왼쪽을 보고 운동을 할 때 바닥과 몸의 각도 코사인 구하기
                left_direction_cos = get_cosine(right_hip_to_ankle, [-1, 0])
                # 오른쪽을 보고 운동을 할 때 바닥과 몸의 각도 코사인 구하기
                right_direction_cos = get_cosine(left_hip_to_ankle, [1, 0])
                
                # 어께와 손목 사이의 거리 구하기
                left_arm_len = pose_preds[0][9][1] - pose_preds[0][5][1]
                right_arm_len = pose_preds[0][10][1] - pose_preds[0][6][1]
                
                if pose_preds[0][0][0] > pose_preds[0][12][0]: # 코가 오른쪽 엉덩이 위치보다 왼쪽에 있으면 is_left = True
                    is_left = True
                else:
                    is_left = False
                    
                if chk == 0:
                    if (is_left is True and left_direction_cos > math.cos(math.pi/3)) or (is_left is False and right_direction_cos > math.cos(math.pi/3)):
                        chk = 1
                elif chk == 1:
                    # 어께-손목 사이의 최대거리 갱신
                    if is_left is True:
                        pushup_max_len = max(pushup_max_len, right_arm_len)
                    else:
                        pushup_max_len = max(pushup_max_len, left_arm_len)
                    
                    if (is_left is True and right_arm_len < pushup_max_len * 0.7) or (is_left is False and left_arm_len < pushup_max_len * 0.7):
                        chk = 2
                    elif (is_left is True and left_direction_cos <= math.cos(math.pi/3)) or (is_left is False and right_direction_cos <= math.cos(math.pi/3)):
                        chk = 0
                else:
                    if (is_left is True and right_arm_len >= pushup_max_len * 0.8) or (is_left is False and left_arm_len >= pushup_max_len * 0.8):
                        chk = 1
                        count  = True
                    elif (is_left is True and left_direction_cos <= math.cos(math.pi/3)) or (is_left is False and right_direction_cos <= math.cos(math.pi/3)):
                        chk = 0
                
            elif work_out == PULL_UP:
                # 팔꿈치->어께 벡터, 팔꿈치->손목 벡터 구하기
                left_elbow_to_shoulder = [pose_preds[0][5][0] - pose_preds[0][7][0], pose_preds[0][5][1] - pose_preds[0][7][1]]
                right_elbow_to_shoulder = [pose_preds[0][6][0] - pose_preds[0][8][0], pose_preds[0][6][1] - pose_preds[0][8][1]]
                left_elbow_to_wrist = [pose_preds[0][9][0] - pose_preds[0][7][0], pose_preds[0][9][1] - pose_preds[0][7][1]]
                right_elbow_to_wrist = [pose_preds[0][10][0] - pose_preds[0][8][0], pose_preds[0][10][1] - pose_preds[0][8][1]]
                left_elbow_cos = get_cosine(left_elbow_to_shoulder, left_elbow_to_wrist)
                right_elbow_cos = get_cosine(right_elbow_to_shoulder, right_elbow_to_wrist)
                
                if chk == 0:
                    # print("in chk 0")
                    if ((pose_preds[0][9][1] < pose_preds[0][5][1]) and (pose_preds[0][10][1] < pose_preds[0][6][1])) and \
                        ((left_elbow_cos < math.cos(math.pi*2/3)) and (right_elbow_cos < math.cos(math.pi*2/3))): # 양 손목이 어께보다 위로 올라가고 양 팔꿈치 각도가 120도 이상이면 준비된 동작으로 간주
                        chk = 1
                elif chk == 1:
                    # print("in chk 1")
                    if left_elbow_cos >= pullup_threshold1 and right_elbow_cos >= pullup_threshold1: # 팔 각도가 90도보다 작아진다면 상태2로 변경
                        chk = 2
                    elif (pose_preds[0][9][1] >= pose_preds[0][5][1]) or (pose_preds[0][10][1] >= pose_preds[0][6][1]): # 양 손목중 하나라도 어께보다 아래에 있다면 운동 준비 동작에서 벗어난 것으로 간주
                        chk = 0
                else:
                    # print("in chk 2")
                    if left_elbow_cos < pullup_threshold2 and right_elbow_cos < pullup_threshold2: # 팔 각도가 100도보다 커진다면 count up
                        chk = 1
                        count  = True
                    elif (pose_preds[0][9][1] >= pose_preds[0][5][1]) or (pose_preds[0][10][1] >= pose_preds[0][6][1]): # 양 손목중 하나라도 어께보다 아래에 있다면 운동 준비 동작에서 벗어난 것으로 간주
                        chk = 0
            elif work_out == SQUAT:
                # cv2.circle(image_bgr, (int(pose_preds[0][5][0]), int(pose_preds[0][5][1])), 6, (CocoColors[5]), -1)
                # cv2.circle(image_bgr, (int(pose_preds[0][6][0]), int(pose_preds[0][6][1])), 6, (CocoColors[6]), -1)
                # cv2.circle(image_bgr, (int(pose_preds[0][7][0]), int(pose_preds[0][7][1])), 6, (CocoColors[7]), -1)
                # cv2.circle(image_bgr, (int(pose_preds[0][8][0]), int(pose_preds[0][8][1])), 6, (CocoColors[8]), -1)
                # cv2.circle(image_bgr, (int(pose_preds[0][9][0]), int(pose_preds[0][9][1])), 6, (CocoColors[9]), -1)
                # cv2.circle(image_bgr, (int(pose_preds[0][10][0]), int(pose_preds[0][10][1])), 6, (CocoColors[10]), -1)
                
                # cv2.line(image_bgr, (int(pose_preds[0][5][0]), int(pose_preds[0][5][1])), (int(pose_preds[0][7][0]), int(pose_preds[0][7][1])), CocoColors[6], 2)
                # cv2.line(image_bgr, (int(pose_preds[0][6][0]), int(pose_preds[0][6][1])), (int(pose_preds[0][8][0]), int(pose_preds[0][8][1])), CocoColors[6], 2)
                # cv2.line(image_bgr, (int(pose_preds[0][7][0]), int(pose_preds[0][7][1])), (int(pose_preds[0][9][0]), int(pose_preds[0][9][1])), CocoColors[6], 2)
                # cv2.line(image_bgr, (int(pose_preds[0][8][0]), int(pose_preds[0][8][1])), (int(pose_preds[0][10][0]), int(pose_preds[0][10][1])), CocoColors[6], 2)
                # cv2.line(image_bgr, (int(pose_preds[0][5][0]), int(pose_preds[0][5][1])), (int(pose_preds[0][6][0]), int(pose_preds[0][6][1])), CocoColors[6], 2)
                # cv2.line(image_bgr, (int(pose_preds[0][5][0]), int(pose_preds[0][5][1])), (int(pose_preds[0][11][0]), int(pose_preds[0][11][1])), CocoColors[6], 2)
                # cv2.line(image_bgr, (int(pose_preds[0][6][0]), int(pose_preds[0][6][1])), (int(pose_preds[0][12][0]), int(pose_preds[0][12][1])), CocoColors[6], 2)
                
                # # 엉덩이, 무릎, 발목 점 찍어주기
                # cv2.circle(image_bgr, (int(pose_preds[0][11][0]), int(pose_preds[0][11][1])), 6, (CocoColors[11]), -1)
                # cv2.circle(image_bgr, (int(pose_preds[0][12][0]), int(pose_preds[0][12][1])), 6, (CocoColors[12]), -1)
                # cv2.circle(image_bgr, (int(pose_preds[0][13][0]), int(pose_preds[0][13][1])), 6, (CocoColors[13]), -1)
                # cv2.circle(image_bgr, (int(pose_preds[0][14][0]), int(pose_preds[0][14][1])), 6, (CocoColors[14]), -1)
                # cv2.circle(image_bgr, (int(pose_preds[0][15][0]), int(pose_preds[0][15][1])), 6, (CocoColors[15]), -1)
                # cv2.circle(image_bgr, (int(pose_preds[0][16][0]), int(pose_preds[0][16][1])), 6, (CocoColors[16]), -1)
                
                # cv2.line(image_bgr, (int(pose_preds[0][11][0]), int(pose_preds[0][11][1])), (int(pose_preds[0][13][0]), int(pose_preds[0][13][1])), CocoColors[6], 2)
                # cv2.line(image_bgr, (int(pose_preds[0][12][0]), int(pose_preds[0][12][1])), (int(pose_preds[0][14][0]), int(pose_preds[0][14][1])), CocoColors[6], 2)
                # cv2.line(image_bgr, (int(pose_preds[0][13][0]), int(pose_preds[0][13][1])), (int(pose_preds[0][15][0]), int(pose_preds[0][15][1])), CocoColors[6], 2)
                # cv2.line(image_bgr, (int(pose_preds[0][14][0]), int(pose_preds[0][14][1])), (int(pose_preds[0][16][0]), int(pose_preds[0][16][1])), CocoColors[6], 2)
                # cv2.line(image_bgr, (int(pose_preds[0][11][0]), int(pose_preds[0][11][1])), (int(pose_preds[0][12][0]), int(pose_preds[0][12][1])), CocoColors[6], 2)

                # 무릎->엉덩이 벡터, 무릎->발목 벡터 구하기
                left_knee_to_hip = [pose_preds[0][11][0] - pose_preds[0][13][0], pose_preds[0][11][1] - pose_preds[0][13][1]]
                right_knee_to_hip = [pose_preds[0][12][0] - pose_preds[0][14][0], pose_preds[0][12][1] - pose_preds[0][14][1]]
                left_knee_to_ankle = [pose_preds[0][15][0] - pose_preds[0][13][0], pose_preds[0][15][1] - pose_preds[0][13][1]]
                right_knee_to_ankle = [pose_preds[0][16][0] - pose_preds[0][14][0], pose_preds[0][16][1] - pose_preds[0][14][1]]
                left_knee_cos = get_cosine(left_knee_to_hip, left_knee_to_ankle)
                right_knee_cos = get_cosine(right_knee_to_hip, right_knee_to_ankle)
                # 5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow, 9: left_wrist, 10: right_wrist
                # 11: left_hip, 12: right_hip, 13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle
                if chk == 0:
                    print("in chk 0")
                    if (left_knee_cos < math.cos(math.pi*5/12)) and (right_knee_cos < math.cos(math.pi*5/12)) and \
                        (pose_preds[0][15][1] > pose_preds[0][13][1]) and (pose_preds[0][16][1] > pose_preds[0][14][1]) and \
                        (pose_preds[0][13][1] > pose_preds[0][11][1]) and (pose_preds[0][14][1] > pose_preds[0][12][1]): # 양 무릎의 각도가 150도 이상이고 엉덩이가 무릎보다 위, 무릎이 발목보다 위면 준비된 동작으로 간주
                        chk = 1
                elif chk == 1:
                    print("in chk 1")
                    if left_knee_cos >= squat_threshold1 and right_knee_cos >= squat_threshold1: # 무릎 각도가 90도보다 작아진다면 상태2로 변경
                        chk = 2
                else:
                    print("in chk 2")
                    if left_knee_cos < squat_threshold2 and right_knee_cos < squat_threshold2: # 무릎 각도가 120도보다 커진다면 count up
                        chk = 1
                        count  = True
        # cv2.imshow('demo',image_bgr)
        # if cv2.waitKey(0) & 0XFF==ord('q'):
        #     cv2.destroyAllWindows()
        
    return count, chk