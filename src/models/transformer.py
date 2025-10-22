import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism"""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear transformations and reshape
        Q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention
        attention, attn_weights = self.attention(Q, K, V, mask)
        
        # Concatenate heads
        attention = attention.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # Final linear transformation
        output = self.w_o(attention)
        
        return output, attn_weights

    def attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attention = torch.matmul(attn_weights, V)
        
        return attention, attn_weights

class FeedForward(nn.Module):
    """Position-wise feed-forward network"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class EncoderLayer(nn.Module):
    """Single encoder layer"""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output, _ = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class DecoderLayer(nn.Module):
    """Single decoder layer"""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # Self-attention with residual connection
        attn_output, _ = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Cross-attention with residual connection
        attn_output, _ = self.cross_attention(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x

class TransformerEncoder(nn.Module):
    """Transformer encoder"""
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, max_len=5000, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout) 
            for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        # Embedding and positional encoding
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoding(x.transpose(0, 1)).transpose(0, 1)
        x = self.dropout(x)
        
        # Pass through encoder layers
        for layer in self.layers:
            x = layer(x, src_mask)
        
        return x

class TransformerDecoder(nn.Module):
    """Transformer decoder"""
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, max_len=5000, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout) 
            for _ in range(n_layers)
        ])
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, encoder_output, src_mask=None, tgt_mask=None):
        # Embedding and positional encoding
        x = self.embedding(tgt) * math.sqrt(self.d_model)
        x = self.pos_encoding(x.transpose(0, 1)).transpose(0, 1)
        x = self.dropout(x)
        
        # Pass through decoder layers
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        
        # Final linear transformation
        output = self.fc_out(x)
        
        return output

class Transformer(nn.Module):
    """Complete Transformer model compatible with your existing interface"""
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, n_heads=8, 
                 n_encoder_layers=6, n_decoder_layers=6, d_ff=2048, max_len=5000, 
                 dropout=0.1, device='cuda'):
        super().__init__()
        
        self.device = device
        self.d_model = d_model
        
        # For compatibility with your existing code
        self.output_dim = tgt_vocab_size
        
        self.encoder = TransformerEncoder(
            src_vocab_size, d_model, n_heads, n_encoder_layers, d_ff, max_len, dropout
        )
        
        self.decoder = TransformerDecoder(
            tgt_vocab_size, d_model, n_heads, n_decoder_layers, d_ff, max_len, dropout
        )

    def create_padding_mask(self, seq, pad_token_id=0):
        """Create padding mask"""
        return (seq != pad_token_id).unsqueeze(1).unsqueeze(2)

    def create_look_ahead_mask(self, size):
        """Create look-ahead mask for decoder"""
        mask = torch.triu(torch.ones(size, size), diagonal=1)
        return mask == 0

    def forward(self, src, tgt, teacher_forcing_ratio=1.0):
        """
        Forward pass compatible with your existing training loop
        
        Args:
            src: Source sequences [batch_size, src_len]
            tgt: Target sequences [batch_size, tgt_len]
            teacher_forcing_ratio: Not used in transformer (always uses teacher forcing during training)
        """
        batch_size = src.size(0)
        tgt_len = tgt.size(1)
        
        # Create masks
        src_mask = self.create_padding_mask(src)
        tgt_padding_mask = self.create_padding_mask(tgt)
        tgt_look_ahead_mask = self.create_look_ahead_mask(tgt_len).to(self.device)
        tgt_mask = tgt_padding_mask & tgt_look_ahead_mask.unsqueeze(0).unsqueeze(0)
        
        # Encoder
        encoder_output = self.encoder(src, src_mask)
        
        # Decoder (excluding the last token for input)
        decoder_input = tgt[:, :-1]
        decoder_output = self.decoder(decoder_input, encoder_output, src_mask, tgt_mask[:, :, :-1, :-1])
        
        # Pad the output to match target length
        padding = torch.zeros(batch_size, 1, self.output_dim).to(self.device)
        output = torch.cat([padding, decoder_output], dim=1)
        
        return output

    def generate(self, src, tokenizer, max_len=50, device='cuda'):
        """
        Generate translation (for inference)
        Compatible with your translate_sentence function
        """
        self.eval()
        
        with torch.no_grad():
            # Encode source
            src_mask = self.create_padding_mask(src)
            encoder_output = self.encoder(src, src_mask)
            
            # Initialize decoder input with BOS token
            decoder_input = torch.tensor([[tokenizer.BOS_ID]], device=device)
            
            for _ in range(max_len):
                tgt_len = decoder_input.size(1)
                tgt_mask = self.create_look_ahead_mask(tgt_len).to(device)
                
                # Decode
                decoder_output = self.decoder(decoder_input, encoder_output, src_mask, tgt_mask.unsqueeze(0).unsqueeze(0))
                
                # Get next token
                next_token_logits = decoder_output[:, -1, :]
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)
                
                # Append to decoder input
                decoder_input = torch.cat([decoder_input, next_token], dim=1)
                
                # Stop if EOS token
                if next_token.item() == tokenizer.EOS_ID:
                    break
            
            return decoder_input.squeeze(0)

# For backward compatibility with your existing code structure
class TransformerSeq2Seq(nn.Module):
    """Wrapper to match your existing Seq2Seq interface"""
    def __init__(self, encoder, decoder, device):
        super().__init__()
        # This is actually a single transformer model
        self.transformer = encoder  # encoder is actually the full transformer
        self.device = device
        self.output_dim = decoder  # decoder is actually vocab_size
        
    def forward(self, src, tgt, teacher_forcing_ratio=1.0):
        return self.transformer(src, tgt, teacher_forcing_ratio)