import torch
import torch.nn as nn
import torch.nn.functional as F

from models.train_rnn_core import BaseConvSequenceModel


class GaWFRNNConv(BaseConvSequenceModel):
    """
    GaWF (Gated with Feedback) RNN Model.
    Encoder and classifier from BaseConvSequenceModel. Forward overridden for feedback.
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
        super(GaWFRNNConv, self).__init__(
            num_classes,
            num_pos,
            kernel_size=kernel_size,
            device=device,
            dropout_rate=dropout_rate,
            hidden_size=hidden_size,
            max_chars=15,
            predict_all_chars=False,
        )
        self.num_classes = num_classes
        self.num_pos = num_pos
        self.hidden_size = hidden_size
        input_size = self.encoder_flatten_size
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        # self._init_recurrent_module(self.rnn)

        feedback_dim = num_classes + num_pos
        combined_weight_size = input_size + hidden_size
        self.U = nn.Parameter(torch.randn(hidden_size, feedback_dim) * 0.01)
        self.V = nn.Parameter(torch.randn(feedback_dim, combined_weight_size) * 0.01)
        self.LNormRNN = nn.LayerNorm(hidden_size)
        self.register_buffer("prev_feedback", None)

        self._init_gawf_params()

    def _init_gawf_params(self) -> None:
        """
        Explicit initialization hook for GaWF-specific parameters (U and V).
        Currently preserves the initialization defined in __init__ to keep behavior unchanged.
        """
        return

    def set_feedback_frozen(self, freeze: bool):
        for p in (self.U, self.V):
            p.requires_grad = not freeze

    def middle_gawf(self, x_t, h_prev, fb_t):
        input_size = x_t.size(-1)
        weight_ih = self.rnn.weight_ih_l0
        weight_hh = self.rnn.weight_hh_l0
        bias_ih = self.rnn.bias_ih_l0
        bias_hh = self.rnn.bias_hh_l0
        V_ih = self.V[:, :input_size].unsqueeze(0)
        V_hh = self.V[:, input_size:].unsqueeze(0)
        trans_ih = torch.matmul(self.U, fb_t * V_ih)
        trans_hh = torch.matmul(self.U, fb_t * V_hh)
        tau = 0.5 # 2.0
        gate_ih = torch.sigmoid(trans_ih / tau) 
        gate_hh = torch.sigmoid(trans_hh / tau) 
        gated_weight_ih = gate_ih * weight_ih.unsqueeze(0)
        gated_weight_hh = gate_hh * weight_hh.unsqueeze(0)
        ih = torch.bmm(x_t.unsqueeze(1), gated_weight_ih.transpose(1, 2)).squeeze(1)
        hh = torch.bmm(h_prev.unsqueeze(1), gated_weight_hh.transpose(1, 2)).squeeze(1)
        if bias_ih is not None:
            ih = ih + bias_ih.unsqueeze(0)
        if bias_hh is not None:
            hh = hh + bias_hh.unsqueeze(0)
        h_t = torch.tanh(ih + hh)
        gated_output = self.LNormRNN(h_t)
        gated_output = F.relu(gated_output)
        return gated_output

    def forward(self, x, use_feedback=True, reset_feedback=False):
        x = x.to(self.device)
        batch_size, frame_num, channels, height, width = x.size()
        x = x.view(batch_size * frame_num, channels, height, width)
        x = self.encoder(x)
        x = x.view(batch_size, frame_num, -1)

        if use_feedback:
            fb_dim = self.num_classes + self.num_pos
            if reset_feedback or self.prev_feedback is None:
                fb = torch.zeros(batch_size, fb_dim, device=x.device, dtype=torch.float32)
            else:
                fb = self.prev_feedback.to(device=x.device, dtype=torch.float32)

            hidden_size = self.rnn.hidden_size
            char_out = torch.empty(batch_size, frame_num, self.num_classes, device=x.device, dtype=x.dtype)
            pos_out = torch.empty(batch_size, frame_num, self.num_pos, device=x.device, dtype=x.dtype)
            h = torch.zeros(batch_size, hidden_size, device=x.device, dtype=x.dtype)

            for t in range(frame_num):
                x_t = x[:, t, :]
                fb_t = fb.clamp(-10, 10).unsqueeze(2)
                gated_output = self.middle_gawf(x_t, h, fb_t)
                gated_output = F.dropout(gated_output, p=0.5, training=self.training)
                char_t, pos_t = self.classifier(gated_output)
                with torch.no_grad():
                    fb = torch.cat([char_t, pos_t], dim=-1)
                h = gated_output
                char_out[:, t, :], pos_out[:, t, :] = char_t, pos_t

            self.prev_feedback = fb.detach()
        else:
            self.prev_feedback = None
            x, _ = self.rnn(x)
            x = self.LNormRNN(x)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)
            char_out, pos_out = self.classifier(x)

        return char_out, pos_out
