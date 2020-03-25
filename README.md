# Dress-detection
TAMU CSCE 636 Prject by Xiangtian Zheng UIN 526001882

## Install dependencies: 

•	Tensorflow-gpu

•	sklearn

•	cv2

•	numpy

•	matplotlib

•	argparse

•	json

•	os


## More tips on running code

Firstly please find “main.py”, which is the main body of this project. It can load data, networks and model. For the model object, it has three functions: “train”, “prediction” and “present”. The first two functions are actually training and test process, whose input can be only video segment of fixed length. The last function of “present” is to output .json and figure, given a video of arbitrary length.

Once finishing “train”, my code will save its model and weights under the folder “output”. When executing “prediction” or “present”, my code will firstly restore the model and weights from that folder. Besides, in the “main” code, I set three flags: “flag_train”, “flag_predict” and “flag_present”, which means if you would like to execute the corresponding function. Because I have already finished training, I turn off “flag_train”. If necessary, you can turn it on. **Note that trained model is saved in the folder "output", including "model.json" and "model.h5". They respectively save the keras model structure and weights.**

If you would like to test my code on new videos, please copy the video to the folder ”example” and then run “present” function.
