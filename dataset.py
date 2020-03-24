import numpy as np
import os
import cv2
import argparse
from sklearn.model_selection import train_test_split

class dress_dataset():
    def __init__(self, hyperpara):
        self.hyperpara = hyperpara
        self.load_all_video()
        self.separate_train_test()
    
    def load_all_video(self):
        self.segment = []
        self.label = []
        for root, dirs, files in os.walk(self.hyperpara.data_dir):
            for file in files:
                if file.endswith("mp4"):
                    print(os.path.join(root, file))
                    video = cv2.VideoCapture(os.path.join(root, file))
                    fps = video.get(cv2.CAP_PROP_FPS)
                    video_array = self.load_single_video(
                        video=video, 
                        fps=fps)
                    video.release()
                    segment, label = self.create_segment(
                        video_array=video_array, 
                        file_name = file)
                    if self.segment==[]:
                        self.segment = np.copy(segment)
                        self.label = np.copy(label.reshape(-1,1))
                    else:
                        self.segment = np.concatenate((self.segment, segment), axis=0)
                        self.label = np.concatenate((self.label, label.reshape(-1,1)), axis=0)
        self.segment = self.segment/255.0
    
    def load_single_video(self, video, fps):
        # Output: video_array_compress: [T, w, l, rgb]
        print(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_ratio = round(fps/self.hyperpara.fps_standard)
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
    
    def create_segment(self, video_array, file_name):
        # Input: video_array [T, w, l, rgb]
        # Output: segment [n, T, w, l, rgb]
        time_step = self.hyperpara.fps_standard*self.hyperpara.time_window
        interval = self.hyperpara.interval
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
    
    def separate_train_test(self):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.segment, self.label, test_size=0.2)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="C:/Users/zheng/Project/SynthPMU/Results/")
    parser.add_argument("--data_dir", type=str, default="./data/")
    parser.add_argument("--fps_standard", type=int, default=10)
    parser.add_argument("--interval", type=int, default=5)
    parser.add_argument("--time_window", type=int, default=2)
    args = parser.parse_args()

    dress_data = dress_dataset(hyperpara = args)
    
    for root,dirs,files in os.walk('./data/'):
        for file in files:
            if file.endswith('mp4'):
                print(os.path.join(root, file))
    
    file_path = "data/class1_case1.mp4"
    video = cv2.VideoCapture(file_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    success, frame = video.read()
    count = 0
    while success:
        success, frame = video.read()
        count += 1
    if success == True:
        cv2.imshow("Frame", frame)