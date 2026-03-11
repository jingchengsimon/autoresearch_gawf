import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseConvSequenceModel(nn.Module):
    """
    Shared encoder (CNN), classifier (fcchar/fcpos), and forward for Conv sequence models
    that use the same pipeline: (B,T,C,H,W) -> encoder -> (B,T,hidden) -> middle -> classifier.
    Subclasses must implement middle(x) and may add extra layers in __init__.
    Encoder output is fixed to (64, 12, 12) feature maps (\"large\" configuration).
    """

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
        super(BaseConvSequenceModel, self).__init__()
        self.device = device
        self.dropout_rate = dropout_rate
        self.max_chars = max_chars
        self.predict_all_chars = predict_all_chars

        # Fixed \"large\" CNN feature configuration: (64, 12, 12) from MP1 output 48x48
        out_ch, out_h, out_w = 64, 12, 12
        mp2_k, mp2_s = 4, 4
        self.conv1 = nn.Conv2d(2, 32, kernel_size=kernel_size, padding="same")
        self.MP1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.LNorm1 = nn.LayerNorm([32, 48, 48])
        self.conv2 = nn.Conv2d(32, out_ch, kernel_size=3, padding=1)
        self.MP2 = nn.MaxPool2d(kernel_size=mp2_k, stride=mp2_s)
        self.LNorm2 = nn.LayerNorm([out_ch, out_h, out_w])

        reduced_ch = max(8, out_ch // 2)
        reduced_h, reduced_w = out_h // 2, out_w // 2
        self.conv_reduce = nn.Conv2d(out_ch, reduced_ch, kernel_size=1)
        self.pool_reduce = nn.AdaptiveAvgPool2d((reduced_h, reduced_w))

        # self.encoder_flatten_size = out_ch * out_h * out_w
        self.encoder_flatten_size = reduced_ch * reduced_h * reduced_w

        if predict_all_chars:
            self.fcchars = nn.Linear(hidden_size, max_chars * num_classes)
            self.fcpos = None
        else:
            self.fcchar = nn.Linear(hidden_size, num_classes)
            self.fcpos = nn.Linear(hidden_size, num_pos)
        self.to(self.device)

    def _init_recurrent_module(self, rnn_module: nn.Module) -> None:
        """
        Initialize recurrent module parameters with a consistent scheme:
        - weight_ih_* with Xavier uniform
        - weight_hh_* with orthogonal
        - bias_* with zeros
        """
        with torch.no_grad():
            for name, param in rnn_module.named_parameters():
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(param)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)

    def encoder(self, x):
        x = self.conv1(x)
        x = self.MP1(x)
        x = self.LNorm1(x)
        x = F.relu(x)
        x = F.dropout2d(x, p=self.dropout_rate, training=self.training)
        x = self.conv2(x)
        x = self.MP2(x)
        x = self.LNorm2(x)

        x = self.conv_reduce(x)
        x = F.relu(x)
        x = self.pool_reduce(x)
        x = F.dropout2d(x, p=self.dropout_rate, training=self.training)
        return x

    def classifier(self, x):
        if self.predict_all_chars:
            chars_out = self.fcchars(x)
            batch_size, frame_num = chars_out.shape[:2]
            num_classes = chars_out.shape[-1] // self.max_chars
            chars_out = chars_out.view(batch_size, frame_num, self.max_chars, num_classes)
            return chars_out, None
        else:
            return self.fcchar(x), self.fcpos(x)

    def middle(self, x):
        raise NotImplementedError("Subclass must implement middle(x)")

    def forward(self, x):
        x = x.to(self.device)
        batch_size, frame_num, channels, height, width = x.size()
        x = x.view(batch_size * frame_num, channels, height, width)
        x = self.encoder(x)
        x = x.view(batch_size, frame_num, -1)
        x = self.middle(x)
        char_out, pos_out = self.classifier(x)
        return char_out, pos_out


class BaseRNNConv(BaseConvSequenceModel):
    """Base class for CNN-RNN models supporting different RNN types."""

    def __init__(
        self,
        num_classes,
        num_pos,
        rnn_class=nn.RNN,
        kernel_size=3,
        device="cuda",
        dropout_rate=0.3,
        hidden_size=256,
        max_chars=15,
        predict_all_chars=False,
    ):
        super(BaseRNNConv, self).__init__(
            num_classes,
            num_pos,
            kernel_size=kernel_size,
            device=device,
            dropout_rate=dropout_rate,
            hidden_size=hidden_size,
            max_chars=max_chars,
            predict_all_chars=predict_all_chars,
        )
        self.rnn = rnn_class(
            input_size=self.encoder_flatten_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        # self._init_recurrent_module(self.rnn)
        self.LNormRNN = nn.LayerNorm(hidden_size)

    def middle(self, x):
        x = self.rnn(x)[0]
        x = self.LNormRNN(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        return x


class RNNConv(BaseRNNConv):
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
        super(RNNConv, self).__init__(
            num_classes,
            num_pos,
            rnn_class=nn.RNN,
            kernel_size=kernel_size,
            device=device,
            dropout_rate=dropout_rate,
            hidden_size=hidden_size,
            max_chars=max_chars,
            predict_all_chars=predict_all_chars,
        )


class GRUConv(BaseRNNConv):
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
        super(GRUConv, self).__init__(
            num_classes,
            num_pos,
            rnn_class=nn.GRU,
            kernel_size=kernel_size,
            device=device,
            dropout_rate=dropout_rate,
            hidden_size=hidden_size,
            max_chars=max_chars,
            predict_all_chars=predict_all_chars,
        )


class LSTMConv(BaseRNNConv):
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
        super(LSTMConv, self).__init__(
            num_classes,
            num_pos,
            rnn_class=nn.LSTM,
            kernel_size=kernel_size,
            device=device,
            dropout_rate=dropout_rate,
            hidden_size=hidden_size,
            max_chars=max_chars,
            predict_all_chars=predict_all_chars,
        )
