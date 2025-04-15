import torch
import torch.nn as nn
import torch.nn.functional as F

import math

class RotaryPositionalEmbeddingBase(nn.Module):
    def __init__(self, d_model, max_len=5000, learnable=False):
        """
        Parameters:
            d_model (int): The model dimension.
            max_len (int): Maximum sequence length.
            learnable (bool): If True, the positional embeddings will be learnable;
                              otherwise, they are fixed buffers.
        """
        super(RotaryPositionalEmbeddingBase, self).__init__()
        assert d_model % 2 == 0, "d_model must be divisible by 2"
        self.d_model = d_model
        
        # Compute the inverse frequency for the embeddings
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        # Create a list of positions
        positions = torch.arange(0, max_len, dtype=torch.float)
        # Compute the outer product to obtain the sinusoid input
        sinusoid_inp = torch.einsum("i,j->ij", positions, inv_freq)
        
        # Precompute sin and cosine embeddings
        sin_embed = torch.sin(sinusoid_inp)
        cos_embed = torch.cos(sinusoid_inp)
        
        if learnable:
            # Make embeddings learnable.
            self.sin_embed = nn.Parameter(sin_embed, requires_grad=True)
            self.cos_embed = nn.Parameter(cos_embed, requires_grad=True)
        else:
            # Register the embeddings as buffers (fixed values).
            self.register_buffer("sin_embed", sin_embed)
            self.register_buffer("cos_embed", cos_embed)

    def rotate_half(self, x):
        """
        Rotates half the dimensions of the input tensor.
        Splits the last dimension into two parts and performs a fixed rotation.
        """
        x_even = -x[..., 1::2]  # Negate every other component
        x_odd  = x[..., 0::2]   # Keep every other component
        
        # Stack the rotated values along a new last dimension
        stacked = torch.stack([x_even, x_odd], dim=-1)
        # Reshape the last two dimensions back into a single dimension
        new_shape = x_even.shape[:-1] + (-1,)
        return stacked.view(*new_shape)
            
    def forward(self, x):
        """
        Applies rotary positional embeddings to input tensor x.
        x is expected to have shape (batch, seq_len, d_model).
        """
        batch, seq_len, d_model = x.size()
        
        # Select the precomputed embeddings up to the sequence length
        sin_embed = self.sin_embed[:seq_len, :].unsqueeze(0)
        cos_embed = self.cos_embed[:seq_len, :].unsqueeze(0)
        
        # Adjust dimensions to match the model's d_model by repeating each embedding along the last dimension
        sin_embed = torch.repeat_interleave(sin_embed, repeats=2, dim=-1)
        cos_embed = torch.repeat_interleave(cos_embed, repeats=2, dim=-1)
        
        # Apply the rotation to the input tensor
        return x * cos_embed + self.rotate_half(x) * sin_embed
class LearnableRotaryPositionalEmbedding(RotaryPositionalEmbeddingBase):
    def __init__(self, d_model, max_len=5000):
        # Set learnable=True for a learnable version
        super(LearnableRotaryPositionalEmbedding, self).__init__(d_model, max_len, learnable=True)

class FixedRotaryPositionalEmbedding(RotaryPositionalEmbeddingBase):
    def __init__(self, d_model, max_len=5000):
        # Use learnable=False to keep embeddings fixed
        super(FixedRotaryPositionalEmbedding, self).__init__(d_model, max_len, learnable=False)




class RotaryChannelEmbeddingBase(nn.Module):
    def __init__(self, c_in, d_model, max_len=5000, learnable=False):
        """
        Base class for Rotary Channel Embeddings that implements the common functionality,
        including the precomputation of sine and cosine embeddings and the forward method.

        Args:
            c_in (int): Number of input channels.
            d_model (int): Model dimension (must be even).
            max_len (int): Maximum sequence length.
            learnable (bool): If True, the sinusoidal embeddings are learnable parameters;
                              otherwise, they are fixed buffers.
        """
        super(RotaryChannelEmbeddingBase, self).__init__()
        assert d_model % 2 == 0, "d_model must be even for ROPE."
        self.d_model = d_model
        self.c_in = c_in

        # Compute the inverse frequency vector.
        # Use 50000 instead of 10000 as in the provided example.
        inv_freq = 1.0 / (50000 ** (torch.arange(0, d_model, 2).float() / d_model))
        # Create a positions vector for max_len positions.
        positions = torch.arange(0, max_len, dtype=torch.float)
        # Compute sinusoidal inputs: shape (max_len, d_model/2)
        sinusoid_inp = torch.einsum("i,j->ij", positions, inv_freq)
        # Pre-compute sine and cosine embeddings.
        sin_embed = torch.sin(sinusoid_inp)
        cos_embed = torch.cos(sinusoid_inp)

        if learnable:
            # Register as learnable parameters.
            self.sin_embed = nn.Parameter(sin_embed, requires_grad=True)
            self.cos_embed = nn.Parameter(cos_embed, requires_grad=True)
        else:
            # Register as buffers (fixed values).
            self.register_buffer("sin_embed", sin_embed)
            self.register_buffer("cos_embed", cos_embed)

    def rotate_half(self, x):
        """
        Helper function that rotates half of the dimensions of x.
        Splits the last dimension into two halves and returns a tensor where
        the first half is replaced with -second half and the second half with the first half.

        Args:
            x: Tensor of shape (batch, seq_len, d_model).
        
        Returns:
            A tensor of shape (batch, seq_len, d_model) with dimensions rotated.
        """
        x_even = -x[..., 1::2]  # Negate the odd-indexed components.
        x_odd  = x[..., 0::2]   # Keep the even-indexed components.

        # Stack along a new last dimension.
        stacked = torch.stack([x_even, x_odd], dim=-1)
        # Merge the last two dimensions.
        return stacked.view(x_even.shape[0], x_even.shape[1], -1)

    def forward(self, x):
        """
        Applies the rotary transformation using pointwise multiplication.

        Args:
            x: Tensor of shape (batch, seq_len, d_model).
            
        Returns:
            Tensor of shape (batch, seq_len, d_model) with rotary embeddings applied.
        """
        batch, seq_len, d_model = x.size()

        # For this implementation, we only use the first 7 positions.
        sin_embed = self.sin_embed[:7, :].unsqueeze(0)  # Shape: (1, 7, d_model/2)
        cos_embed = self.cos_embed[:7, :].unsqueeze(0)  # Shape: (1, 7, d_model/2)

        # Expand embeddings to cover the full d_model by repeating each half.
        sin_embed = torch.repeat_interleave(sin_embed, repeats=2, dim=-1)  # Now shape: (1, 7, d_model)
        cos_embed = torch.repeat_interleave(cos_embed, repeats=2, dim=-1)  # Now shape: (1, 7, d_model)

        # Adjust the time dimension to match the sequence length (assuming seq_len is a multiple of 7).
        repeat_factor = int(seq_len / 7)
        sin_embed = sin_embed.repeat(1, repeat_factor, 1)
        cos_embed = cos_embed.repeat(1, repeat_factor, 1)

        # Apply the rotary transformation.
        return x * cos_embed + self.rotate_half(x) * sin_embed


class RotaryChannelEmbeddingLearnable(RotaryChannelEmbeddingBase):
    def __init__(self, c_in, d_model, max_len=5000):
        """
        Learnable Rotary Channel Embedding: the sinusoidal embeddings are learnable.
        """
        super(RotaryChannelEmbeddingLearnable, self).__init__(c_in, d_model, max_len, learnable=True)


class RotaryChannelEmbeddingFixed(RotaryChannelEmbeddingBase):
    def __init__(self, c_in, d_model, max_len=5000):
        """
        Fixed Rotary Channel Embedding: the sinusoidal embeddings remain fixed.
        """
        super(RotaryChannelEmbeddingFixed, self).__init__(c_in, d_model, max_len, learnable=False)






# # === Example usage ===
# if __name__ == "__main__":
#     batch = 2
#     seq_len = 20
#     d_model = 512  # Must be even
#     dummy_input = torch.zeros(batch, seq_len, d_model)
    
#     rope = RotaryPositionalEmbedding(d_model, max_len=5000)
#     out = rope(dummy_input)
#     print("Output shape:", out.shape)  # Expected: (2, 20, 512)



class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
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


