import cv2
import sys
import numpy
import os

from get_count import predict_image

PUSH_UP = 0
PULL_UP = 1
SQUAT = 2

def main():

    chk = 0 # count를 셀만한 정도로 움직였는지  check / 0: 준비X, 1: 운동 준비O, 2: count를 셀만큼의 가동범위 동작
    count = 0
    
    image = None
    
    # 여기다가 Chatconsumer.receive 함수가 뱉는 imgae_bgr을 image에 넣어서 활용하고 싶은데...

    while image:
        count_chk, chk = predict_image(image, chk, SQUAT)
        if count_chk is True:
            count += 1
            
    print("total count: {}".format(count))

if __name__ == '__main__':
    main()