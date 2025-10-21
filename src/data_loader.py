import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import json
import random 

from tokenizers.spe import SP_BPE
import config


class NMTDataset(Dataset):
    def __init__(self, data_path, tokenizer, source_lang, target_lang,
                data_key='Train', data_ratio=1.0, split=None, split_ratio=0.1, random_seed=41):
        print(f"Loading tokenizer and data for {source_lang}-{target_lang}.")

        self.tokenizer = tokenizer
        pairs = self._load_data(data_path, source_lang, target_lang, data_key)

        if data_ratio < 1.0:
            num = int(len(pairs)*data_ratio)
            pairs=pairs[:num]

            print(f"Scaled down the dataset to {data_ratio*100}% of the dataset.")

        if split in ('train', 'val'):
            if not pairs:
                raise ValueError(f"No data loaded from {data_path}, with key {data_key}.")

            random.seed(random_seed)
            random.shuffle(pairs)

            split_idx = int(len(pairs)*(1-split_ratio))

            if split == 'train':
                self.data = pairs[:split_idx]
            else:
                self.data = pairs[split_idx:]
            
            print(f"Creating '{split}' with {len(self.data)} pairs.")
        
        elif split is None:
            self.data = pairs
            print(f"Using all {len(self.data)} pairs, no split.")
        else:
            raise ValueError('split must be one of None, "train", "val".')

    def _load_data(self, data_path, source_lang, target_lang, data_key):
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        pair_key = f"{source_lang}-{target_lang}"

        if pair_key not in data or data_key not in data[pair_key]:
            raise KeyError(f"Data key '{data_key}' or pair key '{pair_key}' not found in {data_path}.")

        train_data = data[pair_key][data_key]
        
        pairs = []

        for i in train_data.values():
            source = i['source']
            target = i.get('target', None)

            if source:
                pairs.append((source, target))

        print(f"Loaded {len(pairs)} sentence pairs.")
        return pairs

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        source_text, target_text = self.data[idx]
        source_ids = self.tokenizer.encode(source_text, add_special_tokens=True)

        item = {
            'source_ids': torch.tensor(source_ids, dtype=torch.long),
        }

        if target_text is not None:
            target_ids = self.tokenizer.encode(target_text, add_special_tokens=True)
            item['target_ids'] = torch.tensor(target_ids, dtype=torch.long)

        return item

def collate(batch, pad_token_id):
    source_batch, target_batch = [], []
    has_target = 'target_ids' in batch[0]

    for item in batch:
        source_batch.append(item['source_ids'])
        if has_target: 
            target_batch.append(item['target_ids'])

    source_padded = pad_sequence(source_batch, batch_first=True, padding_value=pad_token_id)
    
    item = {
        'source': source_padded
    }

    if has_target:
        target_padded = pad_sequence(target_batch, batch_first=True, padding_value=pad_token_id)
        item['target'] = target_padded

    return item