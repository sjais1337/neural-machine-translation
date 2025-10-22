import torch
from tqdm import tqdm

def train_one_epoch(model, data_loader, optimizer, criterion, device, clip_value=1):
    model.train()
    epoch_loss = 0
    num_batches = 0

    for batch in tqdm(data_loader, desc='Training'):
        try:
            src = batch['source'].to(device, non_blocking=True)
            tgt = batch['target'].to(device, non_blocking=True)

            optimizer.zero_grad()

            outputs = model(src, tgt)
            
            # Calculate loss
            output_dim = outputs.shape[-1]
            outputs_flat = outputs[:, 1:].reshape(-1, output_dim)
            tgt_flat = tgt[:, 1:].reshape(-1)
            
            loss = criterion(outputs_flat, tgt_flat)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            optimizer.step()
            
            # Store loss value and explicitly delete tensors
            batch_loss = loss.item()
            epoch_loss += batch_loss
            num_batches += 1
            
            # Explicit cleanup
            del src, tgt, outputs, outputs_flat, tgt_flat, loss
            
            # Force garbage collection periodically
            if num_batches % 100 == 0:
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
        except Exception as e:
            print(f"Error in batch {num_batches}: {e}")
            # Cleanup on error
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            continue

    return epoch_loss / max(num_batches, 1)


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

def translate_sentence(model, src_tensor, tokenizer, device, max_len=50):
    """
    Universal translation function that works with both LSTM and Transformer models
    """
    model.eval()
    
    # Check if it's a transformer model
    if hasattr(model, 'generate'):
        return translate_sentence_transformer(model, src_tensor, tokenizer, device, max_len)
    else:
        return translate_sentence_lstm(model, src_tensor, tokenizer, device, max_len)

def translate_sentence_lstm(model, src_tensor, tokenizer, device, max_len=50):
    """
    Translates a single source tensor using the model.
    """
    # Ensure src_tensor is a tensor, not a list
    assert isinstance(src_tensor, torch.Tensor), "src_tensor must be a PyTorch tensor"
    
    # Set model to evaluation mode
    model.eval()

    # Get the target <bos> and <eos> token IDs
    trg_bos_id = tokenizer.BOS_ID
    trg_eos_id = tokenizer.EOS_ID
    
    # Ensure src_tensor is on the correct device and has a batch dim [1, src_len]
    if src_tensor.dim() == 1:
        src_tensor = src_tensor.unsqueeze(0)
    src_tensor = src_tensor.to(device)
    
    # 1. Get the encoder context vectors (hidden and cell states)
    with torch.no_grad():
        enc_outs, hidden, cell = model.encoder(src_tensor)
    
    # 2. Start the decoder input with the <bos> token
    trg_indexes = [trg_bos_id]
    
    # 3. Loop for a maximum number of steps
    for i in range(max_len):
        # Get the last predicted token as the new input
        # It needs to be a tensor on the correct device
        trg_tensor = torch.tensor([trg_indexes[-1]], device=device)
        
        with torch.no_grad():
            # Decode one step
            output, hidden, cell = model.decoder(trg_tensor, hidden, cell, enc_outs)
        
        # 4. Get the token with the highest probability
        pred_token = output.argmax(1).item()
        trg_indexes.append(pred_token)

        # 5. If the predicted token is <eos>, stop
        if pred_token == trg_eos_id:
            break
            
    # 6. Use the tokenizer's decode method to convert all IDs back to a string
    trg_text = tokenizer.decode(trg_indexes)
    
    return trg_text

def translate_sentence_transformer(model, src_tensor, tokenizer, device, max_len=50):
    """
    Translation function specifically for Transformer models
    """
    model.eval()
    
    with torch.no_grad():
        if src_tensor.dim() == 1:
            src_tensor = src_tensor.unsqueeze(0)
        src_tensor = src_tensor.to(device)
        
        # Use the generate method of transformer
        result = model.generate(src_tensor, tokenizer, max_len, device)
        return tokenizer.decode(result.tolist())