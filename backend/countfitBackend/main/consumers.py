import json
from channels.generic.websocket import WebsocketConsumer
from asgiref.sync import async_to_sync
import base64
import os, sys
import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))))
from AI.demo.get_count import predict_image

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
import boto3
import argparse

from AI.lib.config import cfg
from AI.lib.config import update_config
from AI.lib.core.function import get_final_preds
from AI.lib.utils.transforms import get_affine_transform
from AI.lib.models.pose_hrnet import get_pose_net

CTX = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

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

class ChatConsumer(WebsocketConsumer):
    def connect(self):
        self.accept()  # 연결 수락
        PUSH_UP = 0
        PULL_UP = 1
        SQUAT = 2
        # 초기 변수 설정

        self.chk = 0
        self.count = 0
        self.exercise_type = SQUAT  # 변경 가능
        #################################### get model ##################################
        # cudnn 관련 설정
        cudnn.benchmark = cfg.CUDNN.BENCHMARK
        torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
        torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

        args = parse_args()
        update_config(cfg, args)

        self.box_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.box_model.to(CTX)
        self.box_model.eval()

        self.pose_model = get_pose_net(cfg, is_train=False)

        # S3에서 모델 파일 다운로드 (필요한 경우)
        aws_key = 'X'
        aws_secret = 'X'
        s3 = boto3.client('s3', aws_access_key_id=aws_key, aws_secret_access_key=aws_secret)
        bucket_name = 'pose-hrnet-path'
        file_name = 'pose_hrnet_w32_384x288.pth'
        local_file_path = os.path.join(os.path.dirname(__file__), file_name)

        if not os.path.isfile(local_file_path):
            s3.download_file(bucket_name, file_name, local_file_path)

        if cfg.TEST.MODEL_FILE:
            self.pose_model.load_state_dict(torch.load(local_file_path, map_location=torch.device('cpu')), strict=False)
        else:
            print('expected model defined in config at TEST.MODEL_FILE')

        self.pose_model = torch.nn.DataParallel(self.pose_model, device_ids=cfg.GPUS)
        self.pose_model.to(CTX)
        self.pose_model.eval()
        #################################### get model ##################################

        # 클라이언트에 연결 확인 메시지 전송
        self.send(text_data=json.dumps({
            'type': 'connection_established',
            'message': 'You are now connected!'
        }))

    def receive(self, text_data):
        text_data_json = json.loads(text_data)
        if text_data_json['type'] == 'video_frame':
            if text_data_json['data']:
                # base64 디코딩
                frame_data = base64.b64decode(text_data_json['data'])
                # numpy 배열로 변환
                np_arr = np.frombuffer(frame_data, np.uint8)
                # OpenCV 이미지로 변환
                image_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                # print(f"image_bgr shape: {image_bgr.shape}") # (640, 320, 3)
                # resized_image = cv2.resize(image_bgr, (320, 240))
                # resized_image = cv2.resize(image_bgr, (160, 120))
                
                # predict_image 함수에 이미지 전달 및 운동 횟수 카운트
                count_chk, self.chk = predict_image(image_bgr, self.chk, self.exercise_type, self.box_model, self.pose_model)
                # count_chk, self.chk = predict_image(resized_image, self.chk, self.exercise_type, self.box_model, self.pose_model)
                if count_chk:
                    self.count += 1
                    print(f"count {self.count}")
                    # 클라이언트에 현재 카운트 전송
                    self.send(text_data=json.dumps({
                        'type': 'count_update',
                        'count': self.count
                    }))

    def disconnect(self, close_code):
        pass