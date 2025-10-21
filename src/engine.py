import torch
import tqdm

def train_one_epoch(model, data_loader, optimizer, criterion, device, clip_value=1):
    model.train()
    epoch_loss = 0

    for batch in tqdm(data_loader, desc='Training'):
        src = batch['source'].to(device)
        tgt = batch['target'].to(device)

        optimizer.zero_grad()

        outputs = model(src, tgt)

        output_dim = outputs.shape[-1]

        outputs_flat = outputs[:, 1:].reshape(-1, output_dim)

        tgt_flat = tgt[:, 1:].reshape(-1)

        loss = criterion(outputs_flat, tgt_flat)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

        optimizer.step()
        
        epoch_loss += loss.item()

    return epoch_loss/len(data_loader)

def evaluate(model, data_loader, criterion, device):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            src = batch['source'].to(device)
            tgt = batch['target'].to(device)

            outputs = model(src, tgt, 0)

            output_dim = outputs.shape[-1]

            outputs_flat = outputs[:, 1:].reshape(-1, output_dim)
            tgt_flat = tgt[:, 1:].reshape(-1)

            loss = criterion(outputs_flat, tgt_flat)

            epoch_loss += loss.item()

    return epoch_loss/len(data_loader)