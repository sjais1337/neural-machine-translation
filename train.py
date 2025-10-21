from torch.utils.data import DataLoader
from functools import partial
import config

from tokenizers.spe import SP_BPE
from src.data_loader import NMTDataset, collate 

tokenizer = SP_BPE.load(config.PREFIX)
collate_fn = partial(collate, pad_token_id=tokenizer.PAD_ID)

print(f"Tokenizer loaded with vocab size: {len(tokenizer.vocab)}")

print("\n--- Loading Training Data ---")
train_dataset = NMTDataset(
    data_path=config.TRAIN_DATA_PATH,
    tokenizer=tokenizer,
    source_lang=config.SOURCE_LANG,
    target_lang=config.TARGET_LANG,
    data_key="Train", 
    split="train",   
    split_ratio=0.1 
)

print("\n--- Loading Validation Data ---")
val_dataset = NMTDataset(
    data_path=config.TRAIN_DATA_PATH, 
    tokenizer=tokenizer,
    source_lang=config.SOURCE_LANG,
    target_lang=config.TARGET_LANG,
    data_key="Train", 
    split="val",      
    split_ratio=0.1 
)

train_loader = DataLoader(
    train_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn
)

val_loader = DataLoader(
    val_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=False, 
    collate_fn=collate_fn
)

print("\n--- Loading Test Data ---")
test_dataset = NMTDataset(
    data_path=config.TEST_DATA_PATH, 
    tokenizer=tokenizer,
    source_lang=config.SOURCE_LANG,
    target_lang=config.TARGET_LANG,
    data_key="Validation",  
    split=None    
)

test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=collate_fn
)
