import time

import numpy as np
import tensorflow as tf

from scr.pose_estimation.pose_net import pose_net


def setup_pose_prediction(cfg):
    tf.reset_default_graph()
    inputs = tf.placeholder(tf.float32, shape=[cfg.batch_size, None, None, 3])
    net_heads = pose_net(cfg).test(inputs)
    outputs = [net_heads['part_prob']]
    if cfg.location_refinement:
        outputs.append(net_heads['locref'])

    restorer = tf.train.Saver()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # Restore variables from disk.
    restorer.restore(sess, cfg.init_weights)

    return sess, inputs, outputs


def extract_cnn_output(outputs_np, cfg):
    ''' extract locref + scmap from network '''
    start = time.time()
    # print("extracting from cnn output" + str(start))
    scmap = outputs_np[0]
    scmap = np.squeeze(scmap)
    locref = None
    # if cfg.location_refinement:
    if True:
        locref = np.squeeze(outputs_np[1])
        shape = locref.shape
        locref = np.reshape(locref, (shape[0], shape[1], -1, 2))
        locref *= cfg.locref_stdev
    # if len(scmap.shape) == 2:  # for single body part!
    # scmap = np.expand_dims(scmap, axis=2)
    # print("done extracting from cnn", str(time.time() - start))
    return scmap, locref


def argmax_pose_predict(scmap, offmat, stride):
    """Combine scoremat and offsets to the final pose."""
    start = time.time()
    maxloc = np.unravel_index(np.argmax(scmap[:, :, 2]),
                              scmap[:, :, 0].shape)
    offset = np.array(offmat[maxloc][0])[::-1]
    pos_f8 = (np.array(maxloc).astype('float') * stride + 0.5 * stride +
              offset)
    np.hstack((pos_f8[::-1],
               [scmap[maxloc][2]]))
    # print("argmax in :" + str(time.time() - start))

    return np.hstack((pos_f8[::-1],
                      [scmap[maxloc][0]]))


def getpose(image, cfg, sess, inputs, outputs, outall=True):
    ''' Extract pose '''
    # print("extract pose" + str(start))
    im = np.expand_dims(image, axis=0).astype(float)
    start = time.time()
    print("session running")
    outputs_np = sess.run(outputs, feed_dict={inputs: im})
    print("session is done in: " + str(time.time() - start))
    scmap, locref = extract_cnn_output(outputs_np, cfg)
    # pose = argmax_pose_predict(scmap, locref, 8.0)
    # print("done extracting pose" + str(time.time() - start))
    if outall:
        return scmap, locref
    else:
        return


###################################################################
# for batchsize bigger than 1


def extract_cnn_outputmulti(output_np, cfg):
    scmap = output_np[0]
    locref = output_np[1]
    shape = locref.shape
    locref = np.reshape(locref, (shape[0], shape[1], shape[2], -1, 2))
    locref *= cfg.locref_stdev
    if len(scmap.shape) == 2:
        scmap = np.expand_dims(scmap, axis=2)
    return scmap, locref


def getPoseNp(cfg, sess, inputs, outputs, image):
    output_np = sess.run(outputs, feed_dict={inputs: image})
    scmap, locref = extract_cnn_outputmulti(output_np, cfg)
    bs, ny, nx, num_joints = scm.shape
