import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class Encoder(nn.Module):
    """The Encoder part of the Seq2Seq model with Attention."""
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(input_dim, emb_dim)
        
        # Use a bi-directional LSTM
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, 
                           dropout=dropout, batch_first=True, bidirectional=True)
        
        self.dropout = nn.Dropout(dropout)
        
        # A fully connected layer to bridge the bidirectional encoder's
        # final hidden state to the decoder's (unidirectional) hidden state.
        self.fc_hidden = nn.Linear(hid_dim * 2, hid_dim)
        self.fc_cell = nn.Linear(hid_dim * 2, hid_dim)

    def forward(self, src):
        # src shape: [batch_size, src_len]
        
        embedded = self.dropout(self.embedding(src))
        # embedded shape: [batch_size, src_len, emb_dim]
        
        # outputs = all hidden states from all time steps (forward and backward)
        # hidden, cell = final hidden/cell state
        outputs, (hidden, cell) = self.rnn(embedded)
        # outputs shape: [batch_size, src_len, hid_dim * 2]
        # hidden shape: [n_layers * 2, batch_size, hid_dim]
        # cell shape: [n_layers * 2, batch_size, hid_dim]
        
        # We need to adapt the final hidden/cell states for the decoder.
        # We concatenate the forward and backward final states from the last layer
        # and pass them through a linear layer.
        
        # Separate the layers
        hidden = hidden.view(self.n_layers, 2, -1, self.hid_dim)
        cell = cell.view(self.n_layers, 2, -1, self.hid_dim)
        
        # Concatenate forward (index 0) and backward (index 1)
        # [n_layers, batch_size, hid_dim * 2]
        hidden_cat = torch.cat((hidden[:, 0, :, :], hidden[:, 1, :, :]), dim=2)
        cell_cat = torch.cat((cell[:, 0, :, :], cell[:, 1, :, :]), dim=2)

        # Pass through the "bridge" linear layers
        hidden = torch.tanh(self.fc_hidden(hidden_cat))
        cell = torch.tanh(self.fc_cell(cell_cat))
        
        # We return all encoder outputs (for attention) and the bridged final states
        return outputs, hidden, cell

class Attention(nn.Module):
    """The Bahdanau Attention mechanism."""
    def __init__(self, hid_dim, n_layers):
        super().__init__()
        
        # We need linear layers to process the encoder outputs and decoder hidden states
        self.attn_hidden = nn.Linear((hid_dim * 2) + (hid_dim * n_layers), hid_dim)
        self.attn_v = nn.Linear(hid_dim, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs):
        # decoder_hidden shape: [n_layers, batch_size, hid_dim]
        # encoder_outputs shape: [batch_size, src_len, hid_dim * 2]
        
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        
        # Reshape decoder_hidden to [batch_size, n_layers * hid_dim]
        decoder_hidden = decoder_hidden.permute(1, 0, 2).reshape(batch_size, -1)
        
        # Repeat decoder hidden state src_len times
        # [batch_size, src_len, n_layers * hid_dim]
        repeated_decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)
        
        # [batch_size, src_len, (hid_dim * 2) + (n_layers * hid_dim)]
        combined_features = torch.cat((repeated_decoder_hidden, encoder_outputs), dim=2)
        
        # Calculate "energy"
        # [batch_size, src_len, hid_dim]
        energy = torch.tanh(self.attn_hidden(combined_features))
        
        # Calculate alignment scores
        # [batch_size, src_len, 1]
        attention = self.attn_v(energy)
        
        # Softmax to get probabilities (weights)
        # [batch_size, src_len]
        return F.softmax(attention.squeeze(2), dim=1)

class Decoder(nn.Module):
    """The Decoder part of the Seq2Seq model with Attention."""
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout, attention):
        super().__init__()
        
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.attention = attention
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        
        # The LSTM input now includes the concatenated embedding AND context vector
        self.rnn = nn.LSTM(emb_dim + (hid_dim * 2), hid_dim, n_layers, 
                           dropout=dropout, batch_first=True)
        
        # The final output layer
        self.fc_out = nn.Linear(emb_dim + hid_dim + (hid_dim * 2), output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, cell, encoder_outputs):
        # input shape: [batch_size]
        # hidden, cell shapes: [n_layers, batch_size, hid_dim]
        # encoder_outputs shape: [batch_size, src_len, hid_dim * 2]
        
        input = input.unsqueeze(1) # [batch_size, 1]
        embedded = self.dropout(self.embedding(input)) # [batch_size, 1, emb_dim]
        
        # 1. Get attention weights
        # a shape: [batch_size, src_len]
        a = self.attention(hidden, encoder_outputs)
        a = a.unsqueeze(1) # [batch_size, 1, src_len]
        
        # 2. Calculate context vector (weighted sum of encoder outputs)
        # torch.bmm is batch matrix multiplication
        # [batch_size, 1, src_len] bmm [batch_size, src_len, hid_dim * 2]
        # context shape: [batch_size, 1, hid_dim * 2]
        context = torch.bmm(a, encoder_outputs)
        
        # 3. Concatenate embedding and context vector
        # [batch_size, 1, emb_dim + (hid_dim * 2)]
        rnn_input = torch.cat((embedded, context), dim=2)
        
        # 4. Feed into the LSTM
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        # output shape: [batch_size, 1, hid_dim]
        
        # 5. Concatenate all relevant vectors for the final prediction
        # (This is a common trick to feed all info to the final layer)
        # [batch_size, 1, emb_dim + hid_dim + (hid_dim * 2)]
        final_input = torch.cat((embedded, output, context), dim=2)
        
        prediction = self.fc_out(final_input.squeeze(1))
        # prediction shape: [batch_size, output_dim]
        
        return prediction, hidden, cell

class Seq2Seq_LSTM(nn.Module):
    """The main wrapper that holds everything together."""
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src shape: [batch_size, src_len]
        # trg shape: [batch_size, trg_len]
        
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim
        
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        
        # 1. Get encoder outputs and initial decoder states
        encoder_outputs, hidden, cell = self.encoder(src)
        
        # 2. First input is <bos> token
        input = trg[:, 0]
        
        # 3. Decoder loop
        for t in range(1, trg_len):
            # Pass encoder_outputs to the decoder at every step
            output, hidden, cell = self.decoder(input, hidden, cell, encoder_outputs)
            
            outputs[:, t] = output
            
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[:, t] if teacher_force else top1
            
        return outputs