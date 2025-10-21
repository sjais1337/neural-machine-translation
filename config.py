import torch
import re 

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TRAIN_DATA_PATH="data/train_data1.json"
TEST_DATA_PATH="data/val_data1.json"
VAL_DATA_PATH=None
PREFIX="nmt"
MODEL_TYPE="LSTM"

SOURCE_LANG='English'
TARGET_LANG='Hindi'

LATIN_REGEX = re.compile(r'[a-zA-Z]')
HINDI_REGEX = re.compile(r'[\u0900-\u097F]')
BENGALI_REGEX = re.compile(r'[\u0980-\u09FF]')

BATCH_SIZE=32
LR=1e-4
NUM_EPOCHS=10

if MODEL_TYPE == 'LSTM':
    pass