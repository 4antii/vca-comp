import torch
from argparse import ArgumentParser
from torch import nn
from ..base import Base

class BandModule(nn.Module):
    """
        Incorporates the condition p either by:
          1. simple concatenation
          2. concatenation after an MLP that maps from cond_dim -> mlp_cond_hidden_dim 
          3. a FiLM layer applied to the RNN output
    """
    def __init__(
        self,
        band_size,
        cond_dim,
        rnn_hidden_size,
        rnn_num_layers,
        rnn_type,
        conditional_layer_type,
        nlp_depth,
        mlp_cond_hidden_dim=64,
    ):
        super(BandModule, self).__init__()
        self.band_size = band_size
        self.conditional_layer_type = conditional_layer_type

        if conditional_layer_type == 2:
            self.cond_transform = nn.Sequential(
                nn.Linear(cond_dim, mlp_cond_hidden_dim),
                nn.ReLU(),
            )
            cond_input_dim = mlp_cond_hidden_dim
        elif conditional_layer_type == 1:
            cond_input_dim = cond_dim
        else:
            cond_input_dim = 0
            self.film_generator = nn.Linear(cond_dim, 2 * rnn_hidden_size)

        rnn_input_dim = band_size + (cond_input_dim if conditional_layer_type in [1, 2] else 0)
        if rnn_type.upper() == 'LSTM':
            self.rnn = nn.LSTM(
                input_size=rnn_input_dim,
                hidden_size=rnn_hidden_size,
                num_layers=rnn_num_layers,
                batch_first=False,
            )
        elif rnn_type.upper() == 'GRU':
            self.rnn = nn.GRU(
                input_size=rnn_input_dim,
                hidden_size=rnn_hidden_size,
                num_layers=rnn_num_layers,
                batch_first=False,
            )
        else:
            raise ValueError("Unsupported rnn_type: choose 'LSTM' or 'GRU'")

        mlp_input_dim = rnn_hidden_size + (cond_input_dim if conditional_layer_type in [1, 2] else 0)

        mlp_layers = []
        current_dim = mlp_input_dim
        for i in range(nlp_depth):
            mlp_layers.append(nn.Linear(current_dim, current_dim))
            mlp_layers.append(nn.PReLU())
        self.mlp = nn.Sequential(*mlp_layers)
        
        self.final_linear = nn.Linear(current_dim, band_size)
        self.out_scale = nn.Parameter(torch.tensor(2.0))

    def forward(self, x, p):
        time_steps, batch, _ = x.size()

        if p is not None and self.conditional_layer_type in [1, 2]:
            p_exp = p.unsqueeze(0).expand(time_steps, batch, -1)
            if self.conditional_layer_type == 2:
                p_exp = self.cond_transform(p_exp)
            rnn_input = torch.cat([x, p_exp], dim=-1)
        else:
            rnn_input = x

        rnn_out, _ = self.rnn(rnn_input)

        if p is not None and self.conditional_layer_type in [1, 2]:
            p_exp2 = p.unsqueeze(0).expand(time_steps, batch, -1)
            if self.conditional_layer_type == 2:
                p_exp2 = self.cond_transform(p_exp2)
            rnn_out = torch.cat([rnn_out, p_exp2], dim=-1)
        elif p is not None and self.conditional_layer_type == 3:
            film_params = self.film_generator(p)
            gamma, beta = film_params.chunk(2, dim=-1)
            gamma = gamma.unsqueeze(0).expand(time_steps, batch, -1)
            beta = beta.unsqueeze(0).expand(time_steps, batch, -1)
            rnn_out = gamma * rnn_out + beta

        mlp_out = self.mlp(rnn_out)
        final_out = self.final_linear(mlp_out)
        mask = torch.sigmoid(final_out) * self.out_scale
        mask = mask.permute(1, 2, 0)
        return mask

class MultibandModelV2(Base):
    """
    STFT-based multiband model with selectable bands, conditional layers, RNN type, and configurable MLP depth
    """
    def __init__(
        self,
        nparams,
        hidden_size=32,
        num_layers=1,
        n_fft=2048,
        hop_length=512,
        num_bands=4,
        rnn_type='LSTM',
        conditional_layer_type=1,
        nlp_depth=3,
        mlp_cond_hidden_dim=64,
        **kwargs
    ):
        super(MultibandModelV2, self).__init__()
        self.save_hyperparameters()

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = torch.hann_window(n_fft)

        self.freq_bins = n_fft // 2 + 1

        base_band_size = self.freq_bins // num_bands
        self.band_sizes = [base_band_size] * num_bands
        remainder = self.freq_bins - base_band_size * num_bands
        if remainder > 0:
            self.band_sizes[-1] += remainder

        band_hidden_size = hidden_size // num_bands

        self.bands = nn.ModuleList()
        for band_size in self.band_sizes:
            self.bands.append(
                BandModule(
                    band_size=band_size,
                    cond_dim=nparams,
                    rnn_hidden_size=band_hidden_size,
                    rnn_num_layers=num_layers,
                    rnn_type=rnn_type,
                    conditional_layer_type=conditional_layer_type,
                    nlp_depth=nlp_depth,
                    mlp_cond_hidden_dim=mlp_cond_hidden_dim
                )
            )

    def forward(self, x, p):
        """
        x: [batch, 1, time]
        p: condition vector [batch, nparams]
        """
        if p is not None:
            if p.dim() == 4:
                p = p.squeeze(0).squeeze(1)
            elif p.dim() == 3:
                p = p.squeeze(1)

        x = x.squeeze(1)

        X = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window.to(x.device),
            return_complex=True,
        )

        magnitude = torch.abs(X)   # [batch, freq, time]
        phase = torch.angle(X)     # [batch, freq, time]
        batch_size, _, time_frames = magnitude.size()

        bands_mag = torch.split(magnitude, self.band_sizes, dim=1)

        masks = []
        for i, band_mag in enumerate(bands_mag):
            # Permute to shape [time, batch, band_size] for the RNN.
            band_input = band_mag.permute(2, 0, 1)
            mask_band = self.bands[i](band_input, p)
            masks.append(mask_band)

        # Combine masks from all bands (along the frequency axis).
        full_mask = torch.cat(masks, dim=1)  # [batch, freq_bins, time]

        masked_magnitude = magnitude * full_mask

        real = masked_magnitude * torch.cos(phase)
        imag = masked_magnitude * torch.sin(phase)
        masked_X = torch.complex(real, imag)

        reconstructed = torch.istft(
            masked_X,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window.to(x.device),
            length=x.size(1),
        )
        return reconstructed.unsqueeze(1)