import numpy as np
import os
import cv2
import argparse
import json

def load_single_video(hyperpara, video, fps):
    # Output: video_array_compress: [T, w, l, rgb]
    print(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_ratio = round(fps/hyperpara.fps_standard)
    success, frame = video.read()
    frame = cv2.resize(frame, (160, 90))
    frame = np.expand_dims(np.array(frame), axis=0)
    video_array = np.copy(frame)
    count = 0
    while success:
        count += 1
        print(count)
        success, frame = video.read()
        if success:
            frame = cv2.resize(frame, (160, 90))
            frame = np.expand_dims(np.array(frame), axis=0)
            video_array = np.concatenate((video_array, frame), axis=0)
    video_array_compress = video_array[0:-1:fps_ratio]
    return video_array_compress
    
def create_segment(hyperpara, video_array, file_name):
    # Input: video_array [T, w, l, rgb]
    # Output: segment [n, T, w, l, rgb]
    time_step = hyperpara.fps_standard*hyperpara.time_window
    interval = hyperpara.present_interval
    for i in range(0, video_array.shape[0]-time_step+1, interval):
        print(i)
        segment_tmp = np.expand_dims(video_array[i:i+time_step], axis=0)
        if i==0:
            segment = np.copy(segment_tmp)
        else:
            segment=np.concatenate((segment, segment_tmp), axis=0)
    label_idx = np.array(range(0, video_array.shape[0]-time_step+1, interval))
    if "class1" in file_name:
        label = np.ones(label_idx.shape[0]).astype(int)
    else:
        label = np.zeros(label_idx.shape[0]).astype(int)
    return segment, label

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)