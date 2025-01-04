import cv2
import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
import streamlit as st 
import torch.nn as nn
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        Args:
            input_dim: Number of channels of input tensor.
            hidden_dim: Number of channels of hidden state.
            kernel_size: Size of the convolutional kernel.
            bias: Whether or not to add the bias.
        """
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2  # To maintain same size
        self.bias = bias

        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,  # For input, forget, cell, and output gates
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        # Combine input and previous hidden state
        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        # Convolution
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

        # Gates
        i = torch.sigmoid(cc_i)  # input gate
        f = torch.sigmoid(cc_f)  # forget gate
        o = torch.sigmoid(cc_o)  # output gate
        g = torch.tanh(cc_g)

        # Cell state
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, spatial_size):
        height, width = spatial_size
        return (
            torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
            torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device)
        )

class ConvLSTM(nn.Module):
    """
    Multi-layer ConvLSTM module.
    """
    def __init__(self, input_dim, hidden_dims, kernel_size, num_layers, batch_first=True, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dims[i - 1]

            cell = ConvLSTMCell(
                input_dim=cur_input_dim,
                hidden_dim=self.hidden_dims[i],
                kernel_size=self.kernel_size[i],
                bias=self.bias
            )
            cell_list.append(cell)

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        input_tensor: 5-D Tensor of shape (batch, seq_len, channels, height, width)
        """
        if not self.batch_first:
            # (seq_len, batch, channels, height, width) -> (batch, seq_len, channels, height, width)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, seq_len, _, h, w = input_tensor.size()

        # Initialize hidden states
        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size=b, spatial_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)

        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []

            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](
                    input_tensor=cur_layer_input[:, t, :, :, :],
                    cur_state=[h, c]
                )
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output  # Input to next layer

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if self.return_all_layers:
            return layer_output_list, last_state_list
        else:
            return layer_output_list[-1], last_state_list[-1]

    def _init_hidden(self, batch_size, spatial_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, spatial_size))
        return init_states

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

class ConvLSTMModel(nn.Module):
    def __init__(self, input_channels=3, hidden_channels=[64, 64, 64], kernel_size=(3, 3), num_layers=3, output_channels=3, output_frames=5):
        super(ConvLSTMModel, self).__init__()
        self.output_frames = output_frames

        self.conv_lstm = ConvLSTM(
            input_dim=input_channels,
            hidden_dims=hidden_channels,
            kernel_size=kernel_size,
            num_layers=num_layers,
            batch_first=True,
            bias=True,
            return_all_layers=False
        )

        # Final convolution layer to map to output channels
        self.conv = nn.Conv2d(
            in_channels=hidden_channels[-1],
            out_channels=output_channels,
            kernel_size=(1, 1),
            padding=0
        )

    def forward(self, input_tensor):
        """
        input_tensor: 5-D Tensor of shape (batch, seq_len, channels, height, width)
        """
        # Pass through ConvLSTM
        lstm_output, _ = self.conv_lstm(input_tensor)
        # lstm_output shape: (batch, seq_len, hidden_channels[-1], height, width)

        # Apply the final convolution to each time step
        outputs = []
        for t in range(lstm_output.size(1)):
            x = lstm_output[:, t, :, :, :]  # Shape: (batch, hidden_channels[-1], height, width)
            x = self.conv(x)  # Shape: (batch, output_channels, height, width)
            outputs.append(x)

        outputs = torch.stack(outputs, dim=1)  # Shape: (batch, seq_len, output_channels, height, width)

        # We need to select the appropriate number of output frames
        return outputs[:, -self.output_frames:, :, :, :]
    

class SpatioTemporalLSTMCell(nn.Module):
    def __init__(self, in_channels, hidden_channels, height, width, filter_size=5, stride=1, layer_norm=True):
        super(SpatioTemporalLSTMCell, self).__init__()

        self.hidden_channels = hidden_channels
        padding = filter_size // 2
        self._forget_bias = 1.0
        self.height = height
        self.width = width

        if layer_norm:
            self.conv_x = nn.Sequential(
                nn.Conv2d(in_channels, hidden_channels * 7, kernel_size=filter_size, stride=stride, padding=padding, bias=False),
                nn.LayerNorm([hidden_channels * 7, height, width])
            )
            self.conv_h = nn.Sequential(
                nn.Conv2d(hidden_channels, hidden_channels * 4, kernel_size=filter_size, stride=stride, padding=padding, bias=False),
                nn.LayerNorm([hidden_channels * 4, height, width])
            )
            self.conv_m = nn.Sequential(
                nn.Conv2d(hidden_channels, hidden_channels * 3, kernel_size=filter_size, stride=stride, padding=padding, bias=False),
                nn.LayerNorm([hidden_channels * 3, height, width])
            )
            self.conv_o = nn.Sequential(
                nn.Conv2d(hidden_channels * 2, hidden_channels, kernel_size=filter_size, stride=stride, padding=padding, bias=False),
                nn.LayerNorm([hidden_channels, height, width])
            )
        else:
            self.conv_x = nn.Conv2d(in_channels, hidden_channels * 7, kernel_size=filter_size, stride=stride, padding=padding, bias=False)
            self.conv_h = nn.Conv2d(hidden_channels, hidden_channels * 4, kernel_size=filter_size, stride=stride, padding=padding, bias=False)
            self.conv_m = nn.Conv2d(hidden_channels, hidden_channels * 3, kernel_size=filter_size, stride=stride, padding=padding, bias=False)
            self.conv_o = nn.Conv2d(hidden_channels * 2, hidden_channels, kernel_size=filter_size, stride=stride, padding=padding, bias=False)

        self.conv_last = nn.Conv2d(hidden_channels * 2, hidden_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x_t, h_t, c_t, m_t):
        x_concat = self.conv_x(x_t)
        h_concat = self.conv_h(h_t)
        m_concat = self.conv_m(m_t)
        i_x, f_x, g_x, i_xp, f_xp, g_xp, o_x = torch.split(x_concat, self.hidden_channels, dim=1)
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.hidden_channels, dim=1)
        i_m, f_m, g_m = torch.split(m_concat, self.hidden_channels, dim=1)

        i_t = torch.sigmoid(i_x + i_h)
        f_t = torch.sigmoid(f_x + f_h + self._forget_bias)
        g_t = torch.tanh(g_x + g_h)

        c_new = f_t * c_t + i_t * g_t

        i_tp = torch.sigmoid(i_xp + i_m)
        f_tp = torch.sigmoid(f_xp + f_m + self._forget_bias)
        g_tp = torch.tanh(g_xp + g_m)

        m_new = f_tp * m_t + i_tp * g_tp

        mem = torch.cat((c_new, m_new), 1)
        o_t = torch.sigmoid(o_x + o_h + self.conv_o(mem))
        h_new = o_t * torch.tanh(self.conv_last(mem))

        return h_new, c_new, m_new

class PredRNN(nn.Module):
    def __init__(self, config):
        super(PredRNN, self).__init__()

        self.config = config
        self.frame_channels = config.img_channels
        self.num_layers = config.num_layers
        self.num_hidden = config.num_hidden
        self.output_frames = config.output_frames

        # Initialize the list of LSTM cells
        cell_list = []
        for i in range(self.num_layers):
            in_channels = self.frame_channels if i == 0 else self.num_hidden[i - 1]
            cell = SpatioTemporalLSTMCell(
                in_channels=in_channels,
                hidden_channels=self.num_hidden[i],
                height=config.img_height,
                width=config.img_width,
                filter_size=config.filter_size,
                stride=config.stride,
                layer_norm=config.layer_norm
            )
            cell_list.append(cell)

        self.cell_list = nn.ModuleList(cell_list)

        # Final convolution to output the generated frames
        self.conv_last = nn.Conv2d(
            self.num_hidden[-1],
            self.frame_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )

    def forward(self, input_frames):
        """
        Forward pass for the RNN.
        - input_frames: [batch_size, input_length, channels, height, width]
        """
        batch_size, seq_length, _, height, width = input_frames.size()

        # Initialize hidden states and memory cells
        h_t = []
        c_t = []
        m_t = torch.zeros(batch_size, self.num_hidden[0], height, width).to(self.config.device)

        for i in range(self.num_layers):
            h_t.append(torch.zeros(batch_size, self.num_hidden[i], height, width).to(self.config.device))
            c_t.append(torch.zeros(batch_size, self.num_hidden[i], height, width).to(self.config.device))

        # To store the predicted frames
        gen_frames = []

        # Process the input sequence frame by frame
        for t in range(seq_length - 1):  # Only use input sequence length for the first part
            x = input_frames[:, t]  # Use the t-th frame from the input sequence
            h_t[0], c_t[0], m_t = self.cell_list[0](x, h_t[0], c_t[0], m_t)  # Compute the next hidden states

            # Propagate the hidden state through the layers
            for i in range(1, self.num_layers):
                h_t[i], c_t[i], m_t = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i], m_t)

            # The predicted frame at time t
            x_gen = self.conv_last(h_t[-1])
            gen_frames.append(x_gen.unsqueeze(1))

        # Concatenate all the generated frames (10 frames)
        gen_frames = torch.cat(gen_frames, dim=1)  # Output shape: [batch_size, 10, channels, height, width]

        # Select frames 6 to 10 (index 5 to 9) as the predicted frames for comparison
        return gen_frames[:, -self.output_frames:, :, :, :]  # Return frames 6-10 as predicted output

class Config:
    def __init__(self):
        self.img_height = 64
        self.img_width = 64
        self.img_channels = 3
        self.input_length = 10
        self.output_frames = 5
        self.total_length = 15                  # input + predicted frames
        self.num_layers = 3
        self.num_hidden = [64, 64, 64]
        self.filter_size = 3                    # Filter Size
        self.stride = 1
        self.layer_norm = False
        self.device = device

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)  # Add batch dimension

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.encoding[:, :seq_len, :].to(x.device)

class VideoTransformer(nn.Module):
    def __init__(self, input_frames, output_frames, d_model=512, nhead=8, num_encoder_layers=3, num_decoder_layers=3, dim_feedforward=2048, frame_size=(64, 64), color_channels=3):
        super(VideoTransformer, self).__init__()
        self.input_frames = input_frames
        self.output_frames = output_frames
        self.d_model = d_model
        self.frame_size = frame_size
        self.color_channels = color_channels

        # Embedding layer for video frames
        self.frame_embed = nn.Linear(color_channels * frame_size[0] * frame_size[1], d_model)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model)

        # Transformer layers
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward),
            num_layers=num_encoder_layers
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward),
            num_layers=num_decoder_layers
        )

        # Output layer to reconstruct frames
        self.output_layer = nn.Linear(d_model, color_channels * frame_size[0] * frame_size[1])

    def forward(self, src, tgt):
        # Flatten and embed input frames
        batch_size, seq_len, channels, height, width = src.size()
        src = src.view(batch_size, seq_len, -1)  # Flatten frames
        src = self.frame_embed(src)

        # Add positional encoding
        src = self.positional_encoding(src)

        # Encode input sequence
        memory = self.encoder(src)

        # Prepare target frames for decoding
        tgt = tgt.view(batch_size, self.output_frames, -1)
        tgt = self.frame_embed(tgt)
        tgt = self.positional_encoding(tgt)

        # Decode output sequence
        output = self.decoder(tgt, memory)

        # Reconstruct frames
        output = self.output_layer(output)
        output = output.view(batch_size, self.output_frames, self.color_channels, self.frame_size[0], self.frame_size[1])
        return output

