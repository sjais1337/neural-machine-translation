import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from functools import partial
import config

from tokenizers.spe import SP_BPE
from src.data_loader import NMTDataset, collate 
import argparse
from src.engine import train_one_epoch, evaluate, translate_sentence
from src.models.model import Encoder, Decoder, Seq2Seq_LSTM

import random

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

    parser.add_argument('--cloud', action='store_true')
    parser.add_argument('--data_ratio',type=float, default=0.1)
    
    args = parser.parse_args()

    train_dataset, test_dataset, val_dataset, \
    train_loader, test_loader, val_loader = initialize_dataset(args.cloud, args.data_ratio)
    
    INPUT_DIM = len(tokenizer.vocab)
    OUTPUT_DIM = len(tokenizer.vocab)
    
    if config.MODEL_TYPE == 'LSTM':
        print("Initializing LSTM model...")
        enc = Encoder(INPUT_DIM, config.EMBEDDING_DIM, config.HIDDEN_DIM, config.NUM_LAYERS, config.DROPOUT)
        dec = Decoder(OUTPUT_DIM, config.EMBEDDING_DIM, config.HIDDEN_DIM, config.NUM_LAYERS, config.DROPOUT)
        model = Seq2Seq_LSTM(enc, dec, config.DEVICE).to(config.DEVICE)
    
    elif config.MODEL_TYPE == 'GRU':
        # from src.models.model import Seq2Seq_GRU
        # model = Seq2Seq_GRU(...)
        print("GRU model not implemented yet.")
        exit()
    else:
        raise ValueError(f"Unknown model type: {config.MODEL_TYPE}")
        
    print(f"Model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")

    # --- 4. Optimizer and Loss Function ---
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.PAD_ID)

    # --- 5. Run the Training Loop ---
    print("\n--- Starting Training ---")
    best_val_loss = float('inf')

    for epoch in range(config.NUM_EPOCHS):
        
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, config.DEVICE)
        val_loss = evaluate(model, val_loader, criterion, config.DEVICE)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'{config.MODEL_TYPE}_best_model.pt')
            print(f"  -> New best model saved!")
        
        random_idx = random.randint(0, len(val_dataset) - 1)
        src_sample, trg_sample = val_dataset.data[random_idx] # Get raw text
        
        # Get the tokenized tensor for the source
        # We fetch it directly from the dataset's __getitem__
        src_tensor = val_dataset[random_idx]['source_ids'] 
        
        # Perform translation
        translated_sentence = translate_sentence(
            model, 
            src_tensor, 
            tokenizer, 
            config.DEVICE
        )
        
        print("-" * 70)
        print(f"EPOCH [{epoch+1}/{config.NUM_EPOCHS}] Example Translation:")
        print(f"  Source (EN): {src_sample}")
        print(f"  Target (HI): {trg_sample}")
        print(f"  Predicted (Model): {translated_sentence}")
        print("-" * 70)
        # ------------------------------------
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'{config.MODEL_TYPE}_best_model.pt')
            print(f"  -> New best model saved!")

        print(f"Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Val. Loss: {val_loss:.3f}")