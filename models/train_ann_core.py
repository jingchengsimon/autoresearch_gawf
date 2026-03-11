import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseMergeConvModel(nn.Module):
    """
    Base for models that adapt ANN to RNN data format by merging frame_num and channels.
    Merges (B, T, C, H, W) -> (B, T*C, H, W); output expanded to (B, T, num_classes) etc.
    Subclasses must implement middle(x) where x is (B, encoder_flatten_size).
    Encoder output is fixed to (64, 12, 12) feature maps (\"large\" configuration).
    """
    def __init__(self, num_classes, num_pos, kernel_size=3, device='cuda', dropout_rate=0.3,
                 hidden_size=256, max_chars=15, predict_all_chars=False):
        super(BaseMergeConvModel, self).__init__()
        self.device = device
        self.dropout_rate = dropout_rate
        self.max_chars = max_chars
        self.predict_all_chars = predict_all_chars
        self._input_channels = None
        self.conv1 = None
        self.MP1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.LNorm1 = None

        # Fixed \"large\" CNN feature configuration: (64, 12, 12) from MP1 output 48x48
        out_ch, out_h, out_w = 64, 12, 12
        mp2_k = 4
        mp2_s = 4
        self.encoder_flatten_size = out_ch * out_h * out_w
        self.conv2 = nn.Conv2d(32, out_ch, kernel_size=3, padding=1)
        self.MP2 = nn.MaxPool2d(kernel_size=mp2_k, stride=mp2_s)
        self.LNorm2 = nn.LayerNorm([out_ch, out_h, out_w])
        if predict_all_chars:
            self.fcchars = nn.Linear(hidden_size, max_chars * num_classes)
            self.fcpos = None
        else:
            self.fcchar = nn.Linear(hidden_size, num_classes)
            self.fcpos = nn.Linear(hidden_size, num_pos)
        self.to(self.device)

    def _ensure_conv1(self, input_channels):
        if self.conv1 is None or self._input_channels != input_channels:
            self._input_channels = input_channels
            self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding='same').to(self.device)
            self.LNorm1 = nn.LayerNorm([32, 48, 48]).to(self.device)

    def encoder(self, x):
        input_channels = x.size(1)
        self._ensure_conv1(input_channels)
        x = self.conv1(x)
        x = self.MP1(x)
        x = self.LNorm1(x)
        x = F.relu(x)
        x = F.dropout2d(x, p=self.dropout_rate, training=self.training)
        x = self.conv2(x)
        x = self.MP2(x)
        x = self.LNorm2(x)
        x = F.relu(x)
        x = F.dropout2d(x, p=self.dropout_rate, training=self.training)
        return x

    def classifier(self, x):
        if self.predict_all_chars:
            chars_out = self.fcchars(x)
            batch_size = chars_out.shape[0]
            num_classes = chars_out.shape[-1] // self.max_chars
            chars_out = chars_out.view(batch_size, self.max_chars, num_classes)
            return chars_out, None
        else:
            return self.fcchar(x), self.fcpos(x)

    def middle(self, x):
        raise NotImplementedError("Subclass must implement middle(x)")

    def forward(self, x):
        x = x.to(self.device)
        batch_size, frame_num, channels, height, width = x.size()
        x = x.view(batch_size, frame_num * channels, height, width)
        x = self.encoder(x)
        x = x.view(batch_size, -1)
        x = self.middle(x)
        char_out, pos_out = self.classifier(x)
        if self.predict_all_chars:
            char_out = char_out.unsqueeze(1).expand(-1, frame_num, -1, -1).contiguous()
        else:
            char_out = char_out.unsqueeze(1).expand(-1, frame_num, -1).contiguous()
            pos_out = pos_out.unsqueeze(1).expand(-1, frame_num, -1).contiguous()
        return char_out, pos_out


class DendriticLayer(nn.Module):
    def __init__(self, input_dim, num_dends, num_soma, dropout=0.0):
        super(DendriticLayer, self).__init__()
        self.num_dends = num_dends
        self.num_soma = num_soma
        self.fc = nn.Linear(input_dim, num_soma * num_dends)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("soma_agg", torch.ones(num_soma, num_dends))

    def forward(self, x):
        x = self.dropout(x)
        x = self.fc(x)
        x = x.view(x.size(0), self.num_soma, self.num_dends)
        x = F.leaky_relu((x * self.soma_agg).sum(dim=2), negative_slope=0.1)
        return x


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.norm(dim=-1, keepdim=True) / (x.shape[-1] ** 0.5)
        return self.scale * x / (norm + self.eps)


class DendriticANN(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers=2, num_dends=32, num_soma=256, dropout=0.5):
        super(DendriticANN, self).__init__()
        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.append(DendriticLayer(in_dim, num_dends, num_soma, dropout=dropout))
            in_dim = num_soma
        self.layers = nn.ModuleList(layers)
        self.out_proj = nn.Linear(num_soma, hidden_size)
        if hidden_size >= 512:
            self.norm = RMSNorm(hidden_size, eps=1e-5)
        else:
            self.norm = nn.LayerNorm(hidden_size, eps=1e-5)
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = hidden_size
        init_gain = 0.5 if hidden_size >= 512 else 1.0
        nn.init.xavier_uniform_(self.out_proj.weight, gain=init_gain)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            if self.hidden_size >= 512:
                if torch.isnan(x).any() or torch.isinf(x).any():
                    x = torch.where(torch.isnan(x) | torch.isinf(x), torch.zeros_like(x), x)
        x = self.out_proj(x)
        x = self.norm(x)
        if self.hidden_size >= 512:
            x = torch.clamp(x, min=-10.0, max=10.0)
            if torch.isnan(x).any() or torch.isinf(x).any():
                x = torch.where(torch.isnan(x) | torch.isinf(x), torch.zeros_like(x), x)
        x = self.dropout(x)
        return x


class DendriticANNConv(BaseMergeConvModel):
    def __init__(
        self,
        num_classes,
        num_pos,
        kernel_size=3,
        device="cuda",
        dropout_rate=0.3,
        hidden_size=256,
        max_chars=15,
        predict_all_chars=False,
        num_layers=2,
        num_dends=32,
        num_soma=256,
    ):
        super(DendriticANNConv, self).__init__(
            num_classes,
            num_pos,
            kernel_size=kernel_size,
            device=device,
            dropout_rate=dropout_rate,
            hidden_size=hidden_size,
            max_chars=max_chars,
            predict_all_chars=predict_all_chars,
        )
        self.dann = DendriticANN(
            input_dim=self.encoder_flatten_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_dends=num_dends,
            num_soma=num_soma,
            dropout=0,
        )

    def middle(self, x):
        return self.dann(x)


class FeedForwardConv(BaseMergeConvModel):
    def __init__(
        self,
        num_classes,
        num_pos,
        kernel_size=3,
        device="cuda",
        dropout_rate=0.3,
        hidden_size=256,
        max_chars=15,
        predict_all_chars=False,
    ):
        ffn_hidden_size = 512 if hidden_size == 256 else hidden_size
        super(FeedForwardConv, self).__init__(
            num_classes,
            num_pos,
            kernel_size=kernel_size,
            device=device,
            dropout_rate=dropout_rate,
            hidden_size=ffn_hidden_size,
            max_chars=max_chars,
            predict_all_chars=predict_all_chars,
        )
        self.fc1 = nn.Linear(self.encoder_flatten_size, ffn_hidden_size)
        self.dropout = nn.Dropout(0.5)

    def middle(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x
