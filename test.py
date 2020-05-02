import tensorflow as tf 
import numpy as np
import functools 
import argparse
import getopt
import os
import cv2
import sys

from network import *
from model import *
from dataset import *
from tools import *

def _arg_define():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="./output/")
    parser.add_argument("--output_dir", type=str, default="./output/")
    parser.add_argument("--data_dir", type=str, default="./data/")
    parser.add_argument("--present_dir", type=str, default="./example/")
    parser.add_argument("--model_saved", type=str, default="./output/model.json")
    parser.add_argument("--weight_saved", type=str, default="./output/model.h5")
    parser.add_argument("--epoch", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--fps_standard", type=int, default=10)
    parser.add_argument("--interval", type=int, default=10)
    parser.add_argument("--present_interval", type=int, default=20)
    parser.add_argument("--time_window", type=int, default=2)
    args = parser.parse_args()
    return args

args = _arg_define()

try:
    opts, args = getopt.getopt(sys.argv[1:], "ho:v", ["video_name"])
except getopt.GetoptError as err:
    print(err)
    sys.exit(2)

for [o, a] in opts:
    if o == "--video_name":
        file = a


#load data
dress_data = dress_dataset(hyperpara = args)

#create a network
dress_net = dressnet(hyperpara=args)

#create a model train and predict
dress_model = dress_detection(
    network = dress_net.model,
    hyperpara = args)

#load specific video and create segment
print(file)
video = cv2.VideoCapture(args.present_dir+file)
fps = video.get(cv2.CAP_PROP_FPS)
video_array = load_single_video(
    hyperpara = args,
    video=video, 
    fps=fps)
video.release()
segment, label = create_segment(
    hyperpara = args,
    video_array=video_array, 
    file_name = file)
#do evaluation and present the results
video_name = os.path.splitext(file)[0]
interval_original = 1.0/fps
ratio = round(fps/args.fps_standard)
interval_new = interval_original*ratio
time_step_start = args.fps_standard*args.time_window
time_step_interval = args.present_interval
time_step_num = segment.shape[0] 
time_step_end = time_step_start+time_step_interval*time_step_num
t = interval_new * np.array(range(time_step_start, time_step_end, time_step_interval))
y_predict = dress_model.present(
    t = t, 
    x_test = segment,
    model_load_name = args.model_saved,
    json_image_save_name = video_name)
