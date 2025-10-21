from torch.utils.data import DataLoader
from functools import partial
import config

from tokenizers.spe import SP_BPE
from src.data_loader import NMTDataset, collate 
import argparse

tokenizer = SP_BPE.load(config.PREFIX)
collate_fn = partial(collate, pad_token_id=tokenizer.PAD_ID)

print(f"Tokenizer loaded with vocab size: {len(tokenizer.vocab)}")

def initialize_dataset(cloud, data_ratio):

    if not 0.0 < data_ratio <= 1.0:
        raise ValueError(f"data_ratio must be between 0.0 and 1.0, got {data_ratio}.")

    train_data_path = config.C_TRAIN_DATA_PATH if cloud else config.TRAIN_DATA_PATH    
    test_data_path = config.C_TEST_DATA_PATH if cloud else config.TEST_DATA_PATH    

    print("\n--- Loading Training Data ---")

    train_dataset = NMTDataset(
        data_path=train_data_path,
        tokenizer=tokenizer,
        source_lang=config.SOURCE_LANG,
        target_lang=config.TARGET_LANG,
        data_key="Train", 
        split="train",   
        split_ratio=0.1,
        data_ratio=data_ratio
    )

    print("\n--- Loading Validation Data ---")
    val_dataset = NMTDataset(
        data_path=train_data_path, 
        tokenizer=tokenizer,
        source_lang=config.SOURCE_LANG,
        target_lang=config.TARGET_LANG,
        data_key="Train", 
        split="val",      
        split_ratio=0.1,
        data_ratio=data_ratio
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
        data_path=test_data_path, 
        tokenizer=tokenizer,
        source_lang=config.SOURCE_LANG,
        target_lang=config.TARGET_LANG,
        data_key="Validation",  
        split=None,
        data_ratio=data_ratio
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn
    )

    return train_dataset, test_dataset, val_dataset, train_loader, test_loader, val_loader

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural Machine Translation pipeline.")

    parser.add_argument('--cloud', type=bool, default=False)
    parser.add_argument('--data_ratio',type=float, default=0.1)
    
    args = parser.parse_args()

    train_dataset, test_dataset, val_dataset, \
    train_loader, test_loader, val_loader = initialize_dataset(args.cloud, args.data_ratio)
