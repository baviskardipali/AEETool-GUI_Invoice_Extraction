"""
Access the config.py folder in the FLASK folder.
This file contains the configurations required excuting the BERT model using Tensorflow.
Any customizations that need to be made can be done by changing the file paths in the config.py
"""
import warnings
from transformers import RobertaTokenizer
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['AUTOGRAPH_VERBOSITY'] = '1'
warnings.filterwarnings("ignore")

MAX_LEN = 400
TRAIN_BATCH_SIZE = 64
VALID_BATCH_SIZE = 16
EPOCHS = 10

# Add your paths here
META_MODEL_PATH_ROBERTA = r"Resource/meta_roberta.bin"
MODEL_PATH_ROBERTA = r"Resource/roberta_tf.h5"
META_MODEL_PATH_BERT = r"Resource/meta_bert-1.bin"
MODEL_PATH_BERT = r"Resource/bert_tf.h5"
MODEL_PATH_DISTILBERT = r"Resource/distilbert_tf-1.h5"
META_MODEL_PATH_DISTILBERT = r"Resource/meta_roberta.bin"
MODEL_PATH_ALBERT = r"Resource/albert_tf.h5"
META_MODEL_PATH_ALBERT = r"Resource/meta_roberta.bin"

#Give a path for poppler for Windows, for Mac keep it commented
POPPLER_PATH = r"C:\Users\shimp\anaconda3\envs\NER-invoices-gui\GUI\Resource\poppler-0.68.0\bin"
