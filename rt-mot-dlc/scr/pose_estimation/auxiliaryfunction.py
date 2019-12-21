import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import ruamel.yaml


def create_config_template():
    """
    Creates a template for config.yaml file. This specific order is preserved while saving as yaml file.
    """
    import ruamel.yaml
    yaml_str = """\
# Project definitions (do not edit)
    Task:
    scorer:
    date:
    \n
# Project path (change when moving around)
    project_path:
    \n
# Annotation data set configuration (and individual video cropping parameters)
    video_sets:
    bodyparts:
    start:
    stop:
    numframes2pick:
    \n
# Plotting configuration
    pcutoff:
    dotsize:
    alphavalue:
    colormap:
    \n
# Training,Evaluation and Analysis configuration
    TrainingFraction:
    iteration:
    resnet:
    snapshotindex:
    batch_size:
    \n
# Cropping Parameters (for analysis and outlier frame detection)
    cropping:
#if cropping is true for analysis, then set the values here:
    x1:
    x2:
    y1:
    y2:
    \n
# Refinement configuration (parameters from annotation dataset configuration also relevant in this stage)
    corner2move2:
    move2corner:
    """
    ruamelFile = ruamel.yaml.YAML()
    cfg_file = ruamelFile.load(yaml_str)
    return (cfg_file, ruamelFile)


def read_config(configname):
    """
    Reads structured config file

    """
    ruamelFile = ruamel.yaml.YAML()
    path = Path(configname)
    with open(path, 'r') as f:
        cfg = ruamelFile.load(f)
    return (cfg)


def attempttomakefolder(foldername, recursive=False):
    ''' Attempts to create a folder with specified name. Does nothing if it already exists. '''

    try:
        os.path.isdir(foldername)
    except TypeError:  # https://www.python.org/dev/peps/pep-0519/
        foldername = os.fspath(foldername)  # https://github.com/AlexEMG/DeepLabCut/issues/105 (windows)

    if os.path.isdir(foldername):
        print(foldername, " already exists!")
    else:
        if recursive:
            os.makedirs(foldername)
        else:
            os.mkdir(foldername)


def SaveData(PredicteData, metadata, dataname, pdindex, save_as_csv):
    ''' Save predicted data as h5 file and metadata as pickle file; created by predict_videos.py '''

    # here is the h5 created
    DataMachine = pd.DataFrame(PredicteData, columns=pdindex, index=range(1))
    DataMachine.to_hdf(dataname, 'df_with_missing', format='table', mode='w')
    if save_as_csv:
        print("Saving csv poses!")
        DataMachine.to_csv(dataname.split('.h5')[0] + '.csv')
    with open(dataname.split('.h5')[0] + 'includingmetadata.pickle', 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(metadata, f, pickle.HIGHEST_PROTOCOL)


## Various functions to get filenames, foldernames etc. based on configuration parameters.

def GetModelFolder(trainFraction, shuffle, cfg):
    Task = cfg['Task']
    date = cfg['date']
    iterate = 'iteration-' + str(cfg['iteration'])
    return Path(
        'dlc-models/' + iterate + '/' + Task + date + '-trainset' + str(int(trainFraction * 100)) + 'shuffle' + str(
            shuffle))


def GetScorerName(cfg, shuffle, trainFraction, trainingsiterations='unknown'):
    ''' Extract the scorer/network name for a particular shuffle, training fraction, etc. '''
    Task = cfg['Task']
    date = cfg['date']
    if trainingsiterations == 'unknown':
        snapshotindex = cfg['snapshotindex']
        if cfg['snapshotindex'] == 'all':
            print(
                "Changing snapshotindext to the last one -- plotting, videomaking, etc. should not be performed for all indices. For more selectivity enter the ordinal number of the snapshot you want (ie. 4 for the fifth).")
            snapshotindex = -1
        else:
            snapshotindex = cfg['snapshotindex']

        modelfolder = os.path.join(cfg["project_path"], str(GetModelFolder(trainFraction, shuffle, cfg)), 'train')
        Snapshots = np.array([fn.split('.')[0] for fn in os.listdir(modelfolder) if "index" in fn])
        increasing_indices = np.argsort([int(m.split('-')[1]) for m in Snapshots])
        Snapshots = Snapshots[increasing_indices]
        # dlc_cfg = read_config(os.path.join(modelfolder,'pose_cfg.yaml'))
        # dlc_cfg['init_weights'] = os.path.join(modelfolder , 'train', Snapshots[snapshotindex])
        SNP = Snapshots[snapshotindex]
        trainingsiterations = (SNP.split(os.sep)[-1]).split('-')[-1]

    scorer = 'DeepCut' + "_resnet" + str(cfg['resnet']) + "_" + Task + str(date) + 'shuffle' + str(shuffle) + '_' + str(
        trainingsiterations)
    return scorer