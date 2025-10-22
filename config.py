import torch
import re 

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TRAIN_DATA_PATH="data/train_data1.json"
TEST_DATA_PATH="data/val_data1.json"
C_TRAIN_DATA_PATH="../drive/MyDrive/train_data1.json"
C_TEST_DATA_PATH="../drive/MyDrive/val_data1.json"
VAL_DATA_PATH=None
PREFIX="nmt"
MODEL_TYPE="TRANSFORMER"


SOURCE_LANG='English'
TARGET_LANG='Hindi'

LATIN_REGEX = re.compile(r'[a-zA-Z]')
HINDI_REGEX = re.compile(r'[\u0900-\u097F]')
BENGALI_REGEX = re.compile(r'[\u0980-\u09FF]')

BATCH_SIZE=16
LEARNING_RATE=1e-4
NUM_EPOCHS=10

DROPOUT = 0.3
MAX_SENTENCE_LEN = 60



if MODEL_TYPE == 'LSTM':
    EMBEDDING_DIM = 256
    HIDDEN_DIM = 512
    NUM_LAYERS = 2
if MODEL_TYPE == 'TRANSFORMER':
    # Transformer hyperparameters
    D_MODEL = 512           # Model dimension
    N_HEADS = 8             # Number of attention heads
    N_ENCODER_LAYERS = 6    # Number of encoder layers
    N_DECODER_LAYERS = 6    # Number of decoder layers
    D_FF = 2048            # Feed-forward dimension
    MAX_LEN = 5000         # Maximum sequence length

    # Existing hyperparameters
    EMBEDDING_DIM = D_MODEL  # For transformer, embedding_dim = d_model
    HIDDEN_DIM = D_MODEL  