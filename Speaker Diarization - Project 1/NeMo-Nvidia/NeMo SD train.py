#!/usr/bin/env python
# coding: utf-8

# In[1]:


# """
# You can run either this notebook locally (if you have all the dependencies and a GPU) or on Google Colab.

# Instructions for setting up Colab are as follows:
# 1. Open a new Python 3 notebook.
# 2. Import this notebook from GitHub (File -> Upload Notebook -> "GITHUB" tab -> copy/paste GitHub URL)
# 3. Connect to an instance with a GPU (Runtime -> Change runtime type -> select "GPU" for hardware accelerator)
# 4. Run this cell to set up dependencies.
# """
# # If you're using Google Colab and not running locally, run this cell.

# # Install dependencies
# !pip install wget
# !apt-get install sox libsndfile1 ffmpeg
# !pip install unidecode

# ## Install NeMo
BRANCH = 'main'
# !python -m pip install git+https://github.com/NVIDIA/NeMo.git@$BRANCH#egg=nemo_toolkit[asr]

# # Install TorchAudio
# !pip install torchaudio>=0.10.0 -f https://download.pytorch.org/whl/torch_stable.html


# In[2]:


import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
# CUDA_VISIBLE_DEVICES options = 0/1/2/3. Make sure to restart kernel


# In[3]:


import os
NEMO_ROOT = os.getcwd()
print(NEMO_ROOT)
import glob
import subprocess
import tarfile
import wget

data_dir = os.path.join(NEMO_ROOT,'data')
os.makedirs(data_dir, exist_ok=True)

# Download the dataset. This will take a few moments...
print("******")
if not os.path.exists(data_dir + '/an4_sphere.tar.gz'):
    an4_url = 'https://dldata-public.s3.us-east-2.amazonaws.com/an4_sphere.tar.gz'  # for the original source, please visit http://www.speech.cs.cmu.edu/databases/an4/an4_sphere.tar.gz 
    an4_path = wget.download(an4_url, data_dir)
    print(f"Dataset downloaded at: {an4_path}")
else:
    print("Tarfile already exists.")
    an4_path = data_dir + '/an4_sphere.tar.gz'

# Untar and convert .sph to .wav (using sox)
tar = tarfile.open(an4_path)
tar.extractall(path=data_dir)

print("Converting .sph to .wav...")
sph_list = glob.glob(data_dir + '/an4/**/*.sph', recursive=True)
for sph_path in sph_list:
    wav_path = sph_path[:-4] + '.wav'
    cmd = ["sox", sph_path, wav_path]
    subprocess.run(cmd)
print("Finished conversion.\n******")


# In[4]:


# !find {data_dir}/an4/wav/an4_clstk  -iname "*.wav" > data/an4/wav/an4_clstk/train_all.txt
# !cat data/an4/wav/an4_clstk/train_all.txt
# -----
# create a list file which has all the wav files with absolute paths for each of the train, dev, and test set

# use headset dataset for now


# In[5]:


# if not os.path.exists('scripts'):
#   print("Downloading necessary scripts")
#   #TODO: change to python
#   !mkdir -p scripts/speaker_tasks
#   !wget -P scripts/speaker_tasks/ https://raw.githubusercontent.com/NVIDIA/NeMo/$BRANCH/scripts/speaker_tasks/filelist_to_manifest.py
# !python {NEMO_ROOT}/scripts/speaker_tasks/filelist_to_manifest.py --filelist {data_dir}/an4/wav/an4_clstk/train_all.txt --id -2 --out {data_dir}/an4/wav/an4_clstk/all_manifest.json --split

# ------

#  convert this text file to a manifest file 
# optionally split the files to train \& dev for evaluating the models during training by using the --split flag
# --id 3 means from last slash 3rd label name of train_all.txt (see script for more info)
# add --split for train and dev
# TODO - test and train manifest are same

# Format:
# manifest file describes a training sample 
# - audio_filepath contains the path to the wav file
# - duration it's duration in seconds, and 
# - label is the speaker class label:
# {"audio_filepath": "<absolute path to dataset>data/an4/wav/an4test_clstk/menk/cen4-menk-b.wav", "duration": 3.9, "label": "menk"}


# In[6]:


# !find {data_dir}/an4/wav/an4test_clstk  -iname "*.wav" > {data_dir}/an4/wav/an4test_clstk/test_all.txt
# !python {NEMO_ROOT}/scripts/speaker_tasks/filelist_to_manifest.py --filelist {data_dir}/an4/wav/an4test_clstk/test_all.txt --id -2 --out {data_dir}/an4/wav/an4test_clstk/test.json
# ---


# In[7]:


# train_manifest = os.path.join(data_dir,'an4/wav/an4_clstk/train.json')
# validation_manifest = os.path.join(data_dir,'an4/wav/an4_clstk/dev.json')
# test_manifest = os.path.join(data_dir,'an4/wav/an4_clstk/dev.json')
# -----------

# NOTE!!! - all labels should be present in train.json (Eg: 2003a is in dev set, still that label should be in train)
# path to manifest
train_manifest = os.path.join(data_dir,'ami_headset/train.json')
validation_manifest = os.path.join(data_dir,'ami_headset/dev.json')
test_manifest = os.path.join(data_dir,'ami_headset/test.json')
print(f"Paths:  \n{train_manifest} \n{validation_manifest} \n{test_manifest}")


# # Training with config
# 
# # train for speaker embeding -> use it on SD

# In[8]:


import nemo
# NeMo's ASR collection - This collection contains complete ASR models and
# building blocks (modules) for ASR
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.models import ClusteringDiarizer
from omegaconf import OmegaConf
from nemo.collections.asr.parts.utils.speaker_utils import rttm_to_labels, labels_to_pyannote_object
# since for evaluation we use pyannote.metrics, convert rttm formats to pyannote Annotation objects


# In[9]:


# The TitaNet model is defined in a config file about:
# 1) model: All arguments that will relate to the Model - preprocessors, encoder, decoder, optimizer and schedulers, datasets etc
# 2) trainer: Any argument to be passed to PyTorch Lightning


MODEL_CONFIG = os.path.join(NEMO_ROOT,'conf/titanet-large.yaml')

# !wget -P conf https://raw.githubusercontent.com/NVIDIA/NeMo/$BRANCH/examples/speaker_tasks/diarization/conf/offline_diarization.yaml
# MODEL_CONFIG = os.path.join(NEMO_ROOT,'conf/offline_diarization.yaml')

config = OmegaConf.load(MODEL_CONFIG)
print(OmegaConf.to_yaml(config))


# In[10]:


# Setting up the train_ds, validation_ds, test_ds datasets and dataloaders within the config
print(OmegaConf.to_yaml(config.model.train_ds))
print(OmegaConf.to_yaml(config.model.validation_ds))


# In[11]:


# add some configs
config.model.train_ds.manifest_filepath = train_manifest
config.model.validation_ds.manifest_filepath = validation_manifest
# config.model.test_ds.manifest_filepath = test_manifest TODO add to to config

config.model.decoder.num_classes = 74
# TODO: change num speaker


# In[12]:


# NeMo models are primarily PyTorch Lightning modules 
import torch
import pytorch_lightning as pl


# In[13]:


print("Trainer config - \n")
print(OmegaConf.to_yaml(config.trainer))


# In[14]:


# Let us modify some trainer configs for this demo
# Checks if we have GPU available and uses it
accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
config.trainer.devices = 4
config.trainer.num_nodes = 4
config.trainer.accelerator = accelerator

# Reduces maximum number of epochs to 5 for quick demonstration
config.trainer.max_epochs = 5

# Remove distributed training flags
config.trainer.strategy = 'dp'

# Remove augmentations
config.model.train_ds.augmentor=None

# init
trainer = pl.Trainer(**config.trainer)


# In[ ]:


# setup the experiment
from nemo.utils.exp_manager import exp_manager
log_dir = exp_manager(trainer, config.get("exp_manager", None))
# The log_dir provides a path to the current logging directory for easy access
print(log_dir)


# In[ ]:


# TitaNet is a speaker embedding extractor model that can be used for speaker identification tasks 
# it generates one label for the entire provided audio stream. Therefore we encapsulate it inside the EncDecSpeakerLabelModel as follows.

speaker_model = nemo_asr.models.EncDecSpeakerLabelModel(cfg=config.model, trainer=trainer)


# In[ ]:


# for training
trainer.fit(speaker_model)


# In[ ]:


# for testing
trainer.test(speaker_model, ckpt_path=None)


# In[ ]:





# In[ ]:





# In[ ]:


# add manifest then
# oracle_model = ClusteringDiarizer(cfg=config)

