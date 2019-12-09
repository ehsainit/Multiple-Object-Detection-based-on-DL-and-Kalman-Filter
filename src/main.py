import os
import os.path
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from skimage.util import img_as_ubyte

import mot
from scr.pose_estimation import auxiliaryfunction
from scr.pose_estimation import predict
from scr.pose_estimation.config import load_config


def analyze_image(config, vid, shuffle=1, trainingsetindex=0, gputouse=None, save_as_csv=False):
    if 'TF_CUDNN_USE_AUTOTUNE' in os.environ:
        del os.environ['TF_CUDNN_USE_AUTOTUNE']  # was potentially set during training

    tf.reset_default_graph()
    start_path = os.getcwd()  # record cwd to return to this directory in the end

    cfg = auxiliaryfunction.read_config(config)
    trainFraction = cfg['TrainingFraction'][trainingsetindex]

    modelfolder = os.path.join(cfg["project_path"], str(auxiliaryfunction.GetModelFolder(trainFraction, shuffle, cfg)))
    path_test_config = Path(modelfolder) / 'test' / 'pose_cfg.yaml'
    try:
        dlc_cfg = load_config(str(path_test_config))
    except FileNotFoundError:
        raise FileNotFoundError(
            "It seems the model for shuffle %s and trainFraction %s does not exist." % (shuffle, trainFraction))

    # Check which snapshots are available and sort them by # iterations
    try:
        Snapshots = np.array(
            [fn.split('.')[0] for fn in os.listdir(os.path.join(modelfolder, 'train')) if "index" in fn])
    except FileNotFoundError:
        raise FileNotFoundError(
            "Snapshots not found! It seems the dataset for shuffle %s has not been trained/does not exist.\n Please train it before using it to analyze videos.\n Use the function 'train_network' to train the network for shuffle %s." % (
                shuffle, shuffle))

    if cfg['snapshotindex'] == 'all':
        print(
            "Snapshotindex is set to 'all' in the config.yaml file. Running video analysis with all snapshots is very costly! Use the function 'evaluate_network' to choose the best the snapshot. For now, changing snapshot index to -1!")
        snapshotindex = -1
    else:
        snapshotindex = cfg['snapshotindex']

    increasing_indices = np.argsort([int(m.split('-')[1]) for m in Snapshots])
    Snapshots = Snapshots[increasing_indices]

    print("Using %s" % Snapshots[snapshotindex], "for model", modelfolder)

    ##################################################
    # Load and setup CNN part detector
    ##################################################

    # Check if data already was generated:
    dlc_cfg['init_weights'] = os.path.join(modelfolder, 'train', Snapshots[snapshotindex])

    # update batchsize (based on parameters in config.yaml)
    dlc_cfg['batch_size'] = 1

    sess, inputs, outputs = predict.setup_pose_prediction(dlc_cfg)

    if gputouse is not None:  # gpu selectinon
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gputouse)
    #####################################################
    # Video analysis
    #####################################################
    print("Starting to analyze % ", vid)
    #####################################################
    # Read Video
    #####################################################
    tracker = mot.Tracker(100)
    vname = Path(vid).stem
    print("Loading ", vid)
    cap = cv2.VideoCapture(vid)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(vname, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 25, (frame_width, frame_height))
    start = time.time()
    while (cap.isOpened()):
        ret, frame = cap.read()
        start = time.time()
        if ret:
            scrmap, locref = getPose(dlc_cfg, sess, inputs, outputs, frame)
            tracker.track(scrmap, locref, 1)
            for fly in range(len(tracker.tracks)):
                x = tracker.tracks[fly].prediction[0]
                y = tracker.tracks[fly].prediction[1]
                # draw a green rectangle to visualize the bounding rect
                cv2.rectangle(frame, (int(x + 10), int(y + 10)), (int(x) - 10, int(y) - 10), tracker.tracks[fly].color,
                              2)
            for d in range(len(tracker.det)):
                x = tracker.det[d][0]
                y = tracker.det[d][1]
                centeroid = (int(x), int(y))
                cv2.circle(frame, centeroid, 4, (255, 255, 9), -1)
            # cv2.imshow('Tracking', frame)
            out.write(frame)
        else:
            print("frame was analyzed in " + str(time.time() - start))
            print("")
            print("")
            break
            # cv2.waitKey(50)
    # do a bit of cleanup
    print("[INFO] cleaning up...")
    cv2.destroyAllWindows()
    out.release()


def getPose(dlc_cfg, sess, inputs, outputs, img):
    frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    frame = img_as_ubyte(frame)
    start = time.time()
    scmap, locref = predict.getpose(frame, dlc_cfg, sess, inputs, outputs)
    # print("score map was extracted in :" + str(time.time() - start))
    return scmap, locref


def getPoseBatchX(cfg, sess, inputs, outputs, image):
    pass


if __name__ == '__main__':
    if len(sys.argv) == 3:
        conf = sys.argv[1]
        vid = sys.argv[2]
        print('configuration :', conf)
        print('input vid :', vid)
        analyze_image(conf, vid)
    else:
        print('Usage: python3 scmapper.py [config path] [image]\n'
              'config path: configaration path of the trained network\n'
              'image: input frame')
