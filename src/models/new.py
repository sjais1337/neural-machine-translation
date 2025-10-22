import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class BiDirectionalEncoder(nn.Module):
    """
    A 2-layer bi-directional LSTM encoder. It reads the source sentence
    forwards and backwards to create a rich context of every word.
    """
    def __init__(self, input_dim, emb_dim, hid_dim, dropout):
        super().__init__()
        
        self.embedding = nn.Embedding(input_dim, emb_dim)
        
        self.rnn = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hid_dim,
            num_layers=2, # As per spec
            dropout=dropout,
            batch_first=True,
            bidirectional=True # The key feature
        )
        
        # A "bridge" layer to adapt the encoder's final state for the decoder
        # It combines the final forward and backward hidden/cell states
        self.fc_hidden = nn.Linear(hid_dim * 2, hid_dim)
        self.fc_cell = nn.Linear(hid_dim * 2, hid_dim)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src shape: [batch_size, src_len]
        
        embedded = self.dropout(self.embedding(src))
        # embedded shape: [batch_size, src_len, emb_dim]
        
        # encoder_outputs contains the concatenated forward and backward hidden
        # states from the top LSTM layer for every time step
        encoder_outputs, (hidden, cell) = self.rnn(embedded)
        # encoder_outputs shape: [batch_size, src_len, hid_dim * 2]
        # hidden shape: [num_layers * 2, batch_size, hid_dim]
        
        # The decoder is not bidirectional, so we need to adapt the context vectors.
        # We concatenate the final forward and backward hidden states from each layer.
        # hidden is stacked [fwd_layer_0, bwd_layer_0, fwd_layer_1, bwd_layer_1, ...]
        # We just need the final layer: hidden[-2,:,:] (final forward) and hidden[-1,:,:] (final backward)
        
        # Concatenate final forward and backward hidden states
        hidden_cat = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        cell_cat = torch.cat((cell[-2,:,:], cell[-1,:,:]), dim=1)
        
        # Pass through the linear "bridge" layers
        decoder_hidden = torch.tanh(self.fc_hidden(hidden_cat))
        decoder_cell = torch.tanh(self.fc_cell(cell_cat))
        
        # The decoder will have 4 layers, so we repeat the context to match
        # [4, batch_size, hid_dim]
        decoder_hidden = decoder_hidden.unsqueeze(0).repeat(4, 1, 1)
        decoder_cell = decoder_cell.unsqueeze(0).repeat(4, 1, 1)
        
        return encoder_outputs, decoder_hidden, decoder_cell

class BahdanauAttention(nn.Module):
    """
    The implementation of Additive (Bahdanau-style) Attention.
    """
    def __init__(self, hid_dim):
        super().__init__()
        
        self.attn_Wa = nn.Linear(hid_dim, hid_dim)
        self.attn_Ua = nn.Linear(hid_dim * 2, hid_dim)
        self.attn_v = nn.Linear(hid_dim, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs):
        # decoder_hidden shape: [batch_size, hid_dim] (from the top layer of the decoder)
        # encoder_outputs shape: [batch_size, src_len, hid_dim * 2]
        
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        
        # Repeat decoder hidden state src_len times to perform calculations
        repeated_decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)
        
        # Calculate energy
        energy = torch.tanh(self.attn_Wa(repeated_decoder_hidden) + self.attn_Ua(encoder_outputs))
        
        # Get alignment scores (attention weights)
        attention = self.attn_v(energy).squeeze(2)
        
        return F.softmax(attention, dim=1)

class DenseResidualDecoder(nn.Module):
    """
    A 4-layer LSTM decoder with Bahdanau attention and dense residual connections.
    """
    def __init__(self, output_dim, emb_dim, hid_dim, dropout, attention):
        super().__init__()
        
        self.output_dim = output_dim
        self.attention = attention
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        
        # The input to the RNN includes the embedding and the context vector from attention
        self.rnn = nn.LSTM(
            (hid_dim * 2) + emb_dim, # Input: concatenated context + embedding
            hid_dim,
            num_layers=4, # As per spec
            dropout=dropout,
            batch_first=True
        )
        
        # DENSE RESIDUAL CONNECTION: The final prediction layer gets input from
        # the embedding, the LSTM output, AND the attention context vector.
        self.fc_out = nn.Linear((hid_dim * 2) + hid_dim + emb_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, cell, encoder_outputs):
        # input shape: [batch_size] (current token)
        # hidden, cell shapes: [4, batch_size, hid_dim]
        # encoder_outputs shape: [batch_size, src_len, hid_dim * 2]
        
        input = input.unsqueeze(1) # [batch_size, 1]
        embedded = self.dropout(self.embedding(input)) # [batch_size, 1, emb_dim]
        
        # Get attention weights using the top-layer hidden state
        # The shape of hidden is [num_layers, batch_size, hid_dim]
        attn_weights = self.attention(hidden[-1], encoder_outputs)
        attn_weights = attn_weights.unsqueeze(1) # [batch_size, 1, src_len]
        
        # Calculate context vector (weighted sum of encoder outputs)
        context = torch.bmm(attn_weights, encoder_outputs)
        # context shape: [batch_size, 1, hid_dim * 2]
        
        # Concatenate embedding and context vector to create RNN input
        rnn_input = torch.cat((embedded, context), dim=2)
        # rnn_input shape: [batch_size, 1, emb_dim + hid_dim * 2]
        
        # Feed into the multi-layer LSTM
        rnn_output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        # rnn_output shape: [batch_size, 1, hid_dim]
        
        # DENSE RESIDUAL CONNECTION
        # Concatenate the original embedding, the LSTM output, and the context vector
        # This creates a "skip connection" for all information to the final layer
        final_input = torch.cat((embedded, rnn_output, context), dim=2)
        
        prediction = self.fc_out(final_input.squeeze(1))
        # prediction shape: [batch_size, output_dim]
        
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    """The main wrapper that holds everything together."""
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim
        
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        
        encoder_outputs, hidden, cell = self.encoder(src)
        
        # First input to the decoder is the <bos> token
        input = trg[:, 0]
        
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell, encoder_outputs)
            outputs[:, t] = output
            
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1) 
            input = trg[:, t] if teacher_force else top1
            
        return outputs