import torch
import torch.nn as nn
import random

class Encoder(nn.Module):
    """The Encoder part of the Seq2Seq model."""
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        # Embedding layer converts token IDs to dense vectors
        self.embedding = nn.Embedding(input_dim, emb_dim)
        
        # LSTM layer processes the sequence of embedded vectors
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        
        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        # src shape: [batch_size, src_len]
        
        # Convert source token IDs to embeddings
        embedded = self.dropout(self.embedding(src))
        # embedded shape: [batch_size, src_len, emb_dim]
        
        # The LSTM returns all hidden states (outputs) and the final hidden/cell state
        outputs, (hidden, cell) = self.rnn(embedded)
        
        # We only need the final hidden and cell states as the context vector
        # hidden shape: [n_layers, batch_size, hid_dim]
        # cell shape: [n_layers, batch_size, hid_dim]
        return hidden, cell

class Decoder(nn.Module):
    """The Decoder part of the Seq2Seq model."""
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        
        # A fully connected layer to map the LSTM output to our vocabulary
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, cell):
        # input shape: [batch_size] (the current token)
        # hidden, cell shapes: [n_layers, batch_size, hid_dim] (from previous step)
        
        # Add a sequence length dimension to the input
        input = input.unsqueeze(1) # shape: [batch_size, 1]
        
        embedded = self.dropout(self.embedding(input)) # shape: [batch_size, 1, emb_dim]
        
        # Process one token through the LSTM
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        # output shape: [batch_size, 1, hid_dim]
        
        # Get the logits for the next token prediction
        prediction = self.fc_out(output.squeeze(1)) # shape: [batch_size, output_dim]
        
        return prediction, hidden, cell

class Seq2Seq_LSTM(nn.Module):
    """The main wrapper for the Seq2Seq model."""
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        """
        Orchestrates the forward pass of the model.
        
        Args:
            src (tensor): The source sentence tensor.
            trg (tensor): The target sentence tensor.
            teacher_forcing_ratio (float): Probability of using the ground truth
                                           target token as the next input.
        """
        # src shape: [batch_size, src_len]
        # trg shape: [batch_size, trg_len]
        
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim
        
        # Tensor to store the decoder's predictions
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        
        # 1. Get the context vectors from the encoder
        hidden, cell = self.encoder(src)
        
        # 2. The first input to the decoder is always the <bos> token
        input = trg[:, 0]
        
        # 3. Loop to generate the rest of the sequence
        for t in range(1, trg_len):
            # Decode one step
            output, hidden, cell = self.decoder(input, hidden, cell)
            
            # Store the prediction
            outputs[:, t] = output
            
            # Decide whether to use "teacher forcing"
            teacher_force = random.random() < teacher_forcing_ratio
            
            # Get the highest predicted token
            top1 = output.argmax(1)
            
            # If teacher forcing, use the actual next token from the target sequence
            # Otherwise, use the model's own prediction
            input = trg[:, t] if teacher_force else top1
            
        return outputs