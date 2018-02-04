from pprint import pformat

import numpy as np
import _pickle as cPickle
import tensorflow as tf
import cv2

from sys import exit
from os import path, listdir, makedirs, environ
from tqdm import tqdm, trange
from params_proto import cli_parse

from helpers import validate_files
from utilities import label_img_to_color

from model import ENet_model
from config import output_dir, run_dir, demo_dir

# environ['CUDA_VISIBLE_DEVICES'] = "2"

@cli_parse
class G:
    model_id = "demo_sequence"
    data_path = "../../datasets/miniscapes-processed/demoVideo/stuttgart_00"
    results_dir = "../../runs/image-segmentation/stuttgart_02"


# load the mean color channels of the train imgs:
train_mean_channels = cPickle.load(open(path.join(output_dir, "mean_channels.pkl"), "rb"))

# load the sequence data:
seq_frame_paths = []
frame_names = sorted(listdir(G.data_path))
for step, frame_name in enumerate(tqdm(frame_names)):
    frame_path = path.join(G.data_path, frame_name)
    seq_frame_paths.append(frame_path)

# validate_files(seq_frame_paths)
# exit()

# define where to place the resulting images:
try:
    makedirs(G.results_dir)
except FileExistsError as e:
    print("output_dir already exist. First remove the existing folder if you want to run this.")
    print(f"output_dir is {G.results_dir}")
    exit()

batch_size = 8
img_height = 512
img_width = 1024

# compute the number of batches needed to iterate through the data:
no_of_frames = len(seq_frame_paths)
no_of_batches = int(no_of_frames / batch_size)

model = ENet_model(G.model_id, img_height=img_height, img_width=img_width,
                   batch_size=batch_size)
no_of_classes = model.no_of_classes

# create a saver for restoring variables/parameters:
saver = tf.train.Saver(tf.trainable_variables(), write_version=tf.train.SaverDef.V2)

with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
    # initialize all variables/parameters:
    init = tf.global_variables_initializer()
    sess.run(init)

    # restore the best trained model:
    saver.restore(sess, path.join(run_dir, "best_model/model_1_epoch_23.ckpt"))

    batch_pointer = 0
    for step in trange(no_of_batches):
        batch_imgs = np.zeros((batch_size, img_height, img_width, 3), dtype=np.float32)
        img_paths = []

        for i in range(batch_size):
            img_path = seq_frame_paths[batch_pointer + i]
            img_paths.append(img_path)

            # read the image:
            img = cv2.imread(img_path, -1)
            img = cv2.resize(img, (img_width, img_height))
            img = img - train_mean_channels
            batch_imgs[i] = img

        batch_pointer += batch_size

        batch_feed_dict = model.create_feed_dict(imgs_batch=batch_imgs,
                                                 early_drop_prob=0.0, late_drop_prob=0.0)

        # run a forward pass and get the logits:
        logits = sess.run(model.logits, feed_dict=batch_feed_dict)

        # save all predicted label images overlayed on the input frames to G.results_dir:
        predictions = np.argmax(logits, axis=3)
        for i in range(batch_size):
            pred_img = predictions[i]
            pred_img_color = label_img_to_color(pred_img)

            img = batch_imgs[i] + train_mean_channels

            img_file_name = img_paths[i].split("/")[-1]
            img_name = img_file_name.split(".png")[0]
            pred_path = path.join(G.results_dir, img_name + "_pred.png")

            overlayed_img = 0.3 * img + 0.7 * pred_img_color

            cv2.imwrite(pred_path, overlayed_img)

        # to make debug easier
        # if step > 2:
        #     break

# create a video of all the resulting overlayed images:
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
movie_path = path.join(G.results_dir, "cityscapes_stuttgart_02_pred.avi")
writer = cv2.VideoWriter(movie_path, fourcc, 20.0, (img_width, img_height))

frame_names = sorted(listdir(G.results_dir))
for step, frame_name in enumerate(tqdm(frame_names)):

    if ".png" in frame_name:
        frame_path = path.join(G.results_dir, frame_name)
        frame = cv2.imread(frame_path, -1)
        if frame is None:
            print(frame_path)

        writer.write(frame)

writer.release()

exit()
