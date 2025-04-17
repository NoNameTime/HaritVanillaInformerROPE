import torch
import torch.nn as nn
import torch.nn.functional as F

import math

class UnifiedRotaryEmbeddingBase(nn.Module):
    def __init__(self,
                 d_model,
                 max_len=6000,
                 learnable=False,
                 mode="positional",  # "positional" or "channel"
                 c_in=None,          # required in "channel" mode
                 fixed_embed_length=None,  # for channel mode: how many positions to precompute (default is 7)
                 base_scale=None):   # denominator base for inverse frequency (default: 10000 for positional, 60000 for channel)
        """
        A unified base class for Rotary Embeddings, supporting both positional and channel applications.
        
        Args:
            d_model (int): Model dimension (must be even).
            max_len (int): Maximum sequence length. In positional mode, this defines the embedding table size.
            learnable (bool): If True, the embeddings are learnable parameters.
            mode (str): Either "positional" or "channel". Determines the forward behavior.
            c_in (int, optional): Number of input channels. Required if mode=="channel".
            fixed_embed_length (int, optional): In channel mode, the number of positions in the precomputed table.
                                                Defaults to 7 if not provided.
            base_scale (float, optional): Scale factor in the denominator for computing the inverse frequency.
                                          Defaults to 10000 for positional and 60000 for channel.
        """
        super(UnifiedRotaryEmbeddingBase, self).__init__()
        assert d_model % 2 == 0, "d_model must be divisible by 2"
        self.d_model = d_model
        self.mode = mode

        if mode == "channel":
            if c_in is None:
                raise ValueError("c_in must be provided for channel mode.")
            self.c_in = c_in
            # In channel mode we use a fixed number of positions from the sinusoidal table.
            self.embed_length = fixed_embed_length if fixed_embed_length is not None else 7
            # Use a different base scale for channel (if not overridden).
            self.base_scale = base_scale if base_scale is not None else 60000
        elif mode == "positional":
            # For positional mode, use the entire sequence up to max_len.
            self.embed_length = max_len
            self.base_scale = base_scale if base_scale is not None else 10000
        else:
            raise ValueError("mode must be either 'positional' or 'channel'")

        # Compute the inverse frequency vector.
        inv_freq = 1.0 / (self.base_scale ** (torch.arange(0, d_model, 2).float() / d_model))
        # Create a positions vector for the precomputed embeddings.
        positions = torch.arange(0, self.embed_length, dtype=torch.float)
        # Compute the sinusoidal input via an outer product.
        sinusoid_inp = torch.einsum("i,j->ij", positions, inv_freq)
        # Precompute sine and cosine embeddings.
        sin_embed = torch.sin(sinusoid_inp)
        cos_embed = torch.cos(sinusoid_inp)
        
        if learnable:
            # Register them as learnable parameters.
            self.sin_embed = nn.Parameter(sin_embed, requires_grad=True)
            self.cos_embed = nn.Parameter(cos_embed, requires_grad=True)
        else:
            # Register as fixed buffers.
            self.register_buffer("sin_embed", sin_embed)
            self.register_buffer("cos_embed", cos_embed)

    def rotate_half(self, x):

        x_even = -x[..., 1::2]  # Negate every odd-indexed element.
        x_odd = x[..., 0::2]    # Keep every even-indexed element.
        # Stack along a new last dimension and then reshape back.
        stacked = torch.stack([x_even, x_odd], dim=-1)
        new_shape = x_even.shape[:-1] + (-1,)
        return stacked.view(*new_shape)

    def forward(self, x):
    
        batch, seq_len, d_model = x.size()

        if self.mode == "positional":
            # Take the first seq_len positions from the precomputed embeddings.
            sin_embed = self.sin_embed[:seq_len, :].unsqueeze(0)  # Shape: (1, seq_len, d_model/2)
            cos_embed = self.cos_embed[:seq_len, :].unsqueeze(0)  # Shape: (1, seq_len, d_model/2)
            # Expand each embedding to cover the full d_model.
            sin_embed = torch.repeat_interleave(sin_embed, repeats=2, dim=-1)  # Shape: (1, seq_len, d_model)
            cos_embed = torch.repeat_interleave(cos_embed, repeats=2, dim=-1)  # Shape: (1, seq_len, d_model)
        elif self.mode == "channel":
            # Use only a fixed number of positions (by default 7) for channel embeddings.
            sin_embed = self.sin_embed[:self.embed_length, :].unsqueeze(0)  # Shape: (1, embed_length, d_model/2)
            cos_embed = self.cos_embed[:self.embed_length, :].unsqueeze(0)  # Shape: (1, embed_length, d_model/2)
            sin_embed = torch.repeat_interleave(sin_embed, repeats=2, dim=-1)  # Now (1, embed_length, d_model)
            cos_embed = torch.repeat_interleave(cos_embed, repeats=2, dim=-1)  # Now (1, embed_length, d_model)
            # The input's sequence length is assumed to be an integer multiple of embed_length.
            repeat_factor = seq_len // self.embed_length
            sin_embed = sin_embed.repeat(1, repeat_factor, 1)
            cos_embed = cos_embed.repeat(1, repeat_factor, 1)
        else:
            raise ValueError("Invalid mode specified. Choose 'positional' or 'channel'.")

        # Apply the rotary transformation.
        return x * cos_embed + self.rotate_half(x) * sin_embed


# Final subclasses for user convenience.

# Rotary Positional Embeddings
class LearnableRotaryPositionalEmbedding(UnifiedRotaryEmbeddingBase):
    def __init__(self, d_model, max_len=6000):
        super(LearnableRotaryPositionalEmbedding, self).__init__(
            d_model=d_model,
            max_len=max_len,
            learnable=True,
            mode="positional"
        )

class FixedRotaryPositionalEmbedding(UnifiedRotaryEmbeddingBase):
    def __init__(self, d_model, max_len=6000):
        super(FixedRotaryPositionalEmbedding, self).__init__(
            d_model=d_model,
            max_len=max_len,
            learnable=False,
            mode="positional"
        )


# Rotary Channel Embeddings
class RotaryChannelEmbeddingLearnable(UnifiedRotaryEmbeddingBase):
    def __init__(self, c_in, d_model, max_len=6000, fixed_embed_length=7):

        super(RotaryChannelEmbeddingLearnable, self).__init__(
            d_model=d_model,
            max_len=max_len,
            learnable=True,
            mode="channel",
            c_in=c_in,
            fixed_embed_length=fixed_embed_length
        )

class RotaryChannelEmbeddingFixed(UnifiedRotaryEmbeddingBase):
    def __init__(self, c_in, d_model, max_len=6000, fixed_embed_length=7):
       
        super(RotaryChannelEmbeddingFixed, self).__init__(
            d_model=d_model,
            max_len=max_len,
            learnable=False,
            mode="channel",
            c_in=c_in,
            fixed_embed_length=fixed_embed_length
        )





# # === Example usage ===
# if __name__ == "__main__":
#     batch = 2
#     seq_len = 20
#     d_model = 512  # Must be even
#     dummy_input = torch.zeros(batch, seq_len, d_model)
    
#     rope = RotaryPositionalEmbedding(d_model, max_len=6000)
#     out = rope(dummy_input)
#     print("Output shape:", out.shape)  # Expected: (2, 20, 512)



class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=6000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.requires_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model, 
                                    kernel_size=3, padding=padding, padding_mode='circular')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1,2)
        return x

class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.requires_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()

class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4; hour_size = 24
        weekday_size = 7; day_size = 32; month_size = 13

        Embed = FixedEmbedding if embed_type=='fixed' else nn.Embedding
        if freq=='t':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)
    
    def forward(self, x):
        x = x.long()
        
        minute_x = self.minute_embed(x[:,:,4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:,:,3])
        weekday_x = self.weekday_embed(x[:,:,2])
        day_x = self.day_embed(x[:,:,1])
        month_x = self.month_embed(x[:,:,0])
        
        return hour_x + weekday_x + day_x + month_x + minute_x

class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h':4, 't':5, 's':6, 'm':1, 'a':1, 'w':2, 'd':3, 'b':3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model)
    
    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = (TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
                                   if embed_type != 'timeF'
                                   else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq))

        # Change the instantiation here:
        # Use the refactored base class's two subclasses:
        self.rpe = LearnableRotaryPositionalEmbedding(d_model=d_model)
        self.rpe_fixed = FixedRotaryPositionalEmbedding(d_model=d_model)

        self.fixed_channel_embedding = RotaryChannelEmbeddingFixed(c_in=c_in, d_model=d_model)
        self.learnable_channel_embedding = RotaryChannelEmbeddingLearnable(c_in=c_in, d_model=d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        # First, apply the value embedding.
        x = self.value_embedding(x)

        # Apply rotary positional embeddings from both learnable and fixed instances.
        # Note: You can call the instances directly, rather than explicitly invoking .forward().
        x = self.rpe(x) + self.rpe_fixed(x)
        
        # Add channel embeddings
        x = self.fixed_channel_embedding(x) + self.learnable_channel_embedding(x)
        #x = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x) #+ self.temporal_embedding(x_mark)
        return self.dropout(x)


