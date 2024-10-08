# Copyright 2022 Baler Contributors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch import nn
from torch.nn import functional as F


import torch.utils.data
from torch.nn import functional as F
from torch.autograd import Function
from ..modules import helper


class LowerBound(Function):
    @staticmethod
    def forward(ctx, inputs, bound):
        b = torch.ones(inputs.size(), device=inputs.device) * bound
        b = b.to(inputs.device)
        ctx.save_for_backward(inputs, b)
        return torch.max(inputs, b)

    @staticmethod
    def backward(ctx, grad_output):
        inputs, b = ctx.saved_tensors

        pass_through_1 = inputs >= b
        pass_through_2 = grad_output < 0

        pass_through = pass_through_1 | pass_through_2
        return pass_through.type(grad_output.dtype) * grad_output, None


class GDN(nn.Module):
    """Generalized divisive normalization layer.
    y[i] = x[i] / sqrt(beta[i] + sum_j(gamma[j, i] * x[j]^2))
    """

    def __init__(
        self,
        ch,
        inverse=False,
        beta_min=1e-6,
        gamma_init=0.1,
        reparam_offset=2**-18,
    ):
        super(GDN, self).__init__()
        self.inverse = inverse
        self.beta_min = beta_min
        self.gamma_init = gamma_init
        self.device = helper.get_device()
        self.reparam_offset = torch.tensor([reparam_offset], device=self.device)

        self.build(ch)

    def build(self, ch):
        self.pedestal = self.reparam_offset**2
        self.beta_bound = (self.beta_min + self.reparam_offset**2) ** 0.5
        self.gamma_bound = self.reparam_offset

        # Create beta param
        beta = torch.sqrt(torch.ones(ch, device=self.device) + self.pedestal)
        self.beta = nn.Parameter(beta)

        # Create gamma param
        eye = torch.eye(ch, device=self.device)
        g = self.gamma_init * eye
        g = g + self.pedestal
        gamma = torch.sqrt(g)
        self.gamma = nn.Parameter(gamma)

    def forward(self, inputs):
        unfold = False
        if inputs.dim() == 5:
            unfold = True
            bs, ch, d, w, h = inputs.size()
            inputs = inputs.view(bs, ch, d * w, h)

        _, ch, _, _ = inputs.size()

        # Beta bound and reparam
        beta = LowerBound.apply(self.beta, self.beta_bound)
        beta = beta**2 - self.pedestal

        # Gamma bound and reparam
        gamma = LowerBound.apply(self.gamma, self.gamma_bound)
        gamma = gamma**2 - self.pedestal
        gamma = gamma.view(ch, ch, 1, 1)

        # Norm pool calc
        norm_ = nn.functional.conv2d(inputs**2, gamma, beta)
        norm_ = torch.sqrt(norm_)

        # Apply norm
        if self.inverse:
            outputs = inputs * norm_
        else:
            outputs = inputs / norm_

        if unfold:
            outputs = outputs.view(bs, ch, d, w, h)
        return outputs


class AE(nn.Module):
    # This class is a modified version of the original class by George Dialektakis found at
    # https://github.com/Autoencoders-compression-anomaly/Deep-Autoencoders-Data-Compression-GSoC-2021
    # Released under the Apache License 2.0 found at https://www.apache.org/licenses/LICENSE-2.0.txt
    # Copyright 2021 George Dialektakis

    def __init__(self, n_features, z_dim, *args, **kwargs):
        super(AE, self).__init__(*args, **kwargs)

        self.activations = {}

        # encoder
        self.en1 = nn.Linear(n_features, 200, dtype=torch.float64)
        self.en2 = nn.Linear(200, 100, dtype=torch.float64)
        self.en3 = nn.Linear(100, 50, dtype=torch.float64)
        self.en4 = nn.Linear(50, z_dim, dtype=torch.float64)
        # decoder
        self.de1 = nn.Linear(z_dim, 50, dtype=torch.float64)
        self.de2 = nn.Linear(50, 100, dtype=torch.float64)
        self.de3 = nn.Linear(100, 200, dtype=torch.float64)
        self.de4 = nn.Linear(200, n_features, dtype=torch.float64)

        self.n_features = n_features
        self.z_dim = z_dim

    def encode(self, x):
        h1 = F.leaky_relu(self.en1(x))
        h2 = F.leaky_relu(self.en2(h1))
        h3 = F.leaky_relu(self.en3(h2))
        return self.en4(h3)

    def decode(self, z):
        h4 = F.leaky_relu(self.de1(z))
        h5 = F.leaky_relu(self.de2(h4))
        h6 = F.leaky_relu(self.de3(h5))
        out = self.de4(h6)
        return out

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

    # Implementation of activation extraction using the forward_hook method

    def get_hook(self, layer_name):
        def hook(model, input, output):
            self.activations[layer_name] = output.detach()

        return hook

    def get_layers(self) -> list:
        return [self.en1, self.en2, self.en3, self.de1, self.de2, self.de3]

    def store_hooks(self) -> list:
        layers = self.get_layers()
        hooks = []
        for i in range(len(layers)):
            hooks.append(layers[i].register_forward_hook(self.get_hook(str(i))))
        return hooks

    def get_activations(self) -> dict:
        for kk in self.activations:
            self.activations[kk] = F.leaky_relu(self.activations[kk])
        return self.activations

    def detach_hooks(self, hooks: list) -> None:
        for hook in hooks:
            hook.remove()


class CFD_dense_AE(nn.Module):
    # This class is a modified version of the original class by George Dialektakis found at
    # https://github.com/Autoencoders-compression-anomaly/Deep-Autoencoders-Data-Compression-GSoC-2021
    # Released under the Apache License 2.0 found at https://www.apache.org/licenses/LICENSE-2.0.txt
    # Copyright 2021 George Dialektakis

    def __init__(self, n_features, z_dim, *args, **kwargs):
        super(CFD_dense_AE, self).__init__(*args, **kwargs)

        self.activations = {}

        # encoder
        self.en1 = nn.Linear(n_features, 200, dtype=torch.float)
        self.en2 = nn.Linear(200, 100, dtype=torch.float)
        self.en3 = nn.Linear(100, 50, dtype=torch.float)
        self.en4 = nn.Linear(50, z_dim, dtype=torch.float)
        # decoder
        self.de1 = nn.Linear(z_dim, 50, dtype=torch.float)
        self.de2 = nn.Linear(50, 100, dtype=torch.float)
        self.de3 = nn.Linear(100, 200, dtype=torch.float)
        self.de4 = nn.Linear(200, n_features, dtype=torch.float)

        self.n_features = n_features
        self.z_dim = z_dim

    def encode(self, x):
        h1 = F.leaky_relu(self.en1(x))
        h2 = F.leaky_relu(self.en2(h1))
        h3 = F.leaky_relu(self.en3(h2))
        return self.en4(h3)

    def decode(self, z):
        h4 = F.leaky_relu(self.de1(z))
        h5 = F.leaky_relu(self.de2(h4))
        h6 = F.leaky_relu(self.de3(h5))
        out = self.de4(h6)
        return out

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

    # Implementation of activation extraction using the forward_hook method

    def get_hook(self, layer_name):
        def hook(model, input, output):
            self.activations[layer_name] = output.detach()

        return hook

    def get_layers(self) -> list:
        return [self.en1, self.en2, self.en3, self.de1, self.de2, self.de3]

    def store_hooks(self) -> list:
        layers = self.get_layers()
        hooks = []
        for i in range(len(layers)):
            hooks.append(layers[i].register_forward_hook(self.get_hook(str(i))))
        return hooks

    def get_activations(self) -> dict:
        for kk in self.activations:
            self.activations[kk] = F.leaky_relu(self.activations[kk])
        return self.activations

    def detach_hooks(self, hooks: list) -> None:
        for hook in hooks:
            hook.remove()


class AE_Dropout_BN(nn.Module):
    def __init__(self, n_features, z_dim, *args, **kwargs):
        super(AE_Dropout_BN, self).__init__(*args, **kwargs)

        # encoder
        self.enc_nn = nn.Sequential(
            nn.Linear(n_features, 200, dtype=torch.float64),
            nn.Dropout(p=0.5),
            nn.LeakyReLU(),
            # nn.BatchNorm1d(200,dtype=torch.float64),
            nn.Linear(200, 100, dtype=torch.float64),
            nn.Dropout(p=0.4),
            nn.LeakyReLU(),
            # nn.BatchNorm1d(100,dtype=torch.float64),
            nn.Linear(100, 50, dtype=torch.float64),
            nn.Dropout(p=0.3),
            nn.LeakyReLU(),
            # nn.BatchNorm1d(50,dtype=torch.float64),
            nn.Linear(50, z_dim, dtype=torch.float64),
            nn.Dropout(p=0.2),
            nn.LeakyReLU(),
            # nn.BatchNorm1d(z_dim,dtype=torch.float64)
        )

        # decoder
        self.dec_nn = nn.Sequential(
            nn.Linear(z_dim, 50, dtype=torch.float64),
            # nn.Dropout(p=0.2),
            nn.LeakyReLU(),
            nn.BatchNorm1d(50, dtype=torch.float64),
            nn.Linear(50, 100, dtype=torch.float64),
            # nn.Dropout(p=0.3),
            nn.LeakyReLU(),
            nn.BatchNorm1d(100, dtype=torch.float64),
            nn.Linear(100, 200, dtype=torch.float64),
            # nn.Dropout(p=0.4),
            nn.LeakyReLU(),
            nn.BatchNorm1d(200, dtype=torch.float64),
            nn.Linear(200, n_features, dtype=torch.float64),
            # nn.Dropout(p=0.5),
            nn.BatchNorm1d(n_features, dtype=torch.float64),
            nn.ReLU(),
        )

        self.n_features = n_features
        self.z_dim = z_dim

    def encode(self, x):
        out = self.enc_nn(x)
        return out

    def decode(self, z):
        out = self.dec_nn(z)
        return out

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)


class Conv_AE(nn.Module):
    def __init__(self, n_features, z_dim, *args, **kwargs):
        super(Conv_AE, self).__init__(*args, **kwargs)

        self.q_z_mid_dim = 2000
        self.q_z_output_dim = 128
        self.conv_op_shape = None

        # Encoder

        # Conv Layers
        self.q_z_conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(2, 5), stride=(1), padding=(1)),
            # nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=(3), stride=(1), padding=(1)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=(3), stride=(1), padding=(0)),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        # Flatten
        self.flatten = nn.Flatten(start_dim=1)

        # Linear layers
        self.q_z_lin = nn.Sequential(
            nn.Linear(self.q_z_output_dim, self.q_z_mid_dim),
            nn.ReLU(),
            # nn.BatchNorm1d(self.q_z_output_dim),
            nn.Linear(self.q_z_mid_dim, z_dim),
            nn.ReLU(),
        )

        # Decoder

        # Linear layers
        self.p_x_lin = nn.Sequential(
            nn.Linear(z_dim, self.q_z_mid_dim),
            nn.ReLU(),
            # nn.BatchNorm1d(self.q_z_output_dim),
            nn.Linear(self.q_z_mid_dim, self.q_z_output_dim),
            nn.ReLU()
            # nn.BatchNorm1d(42720)
        )
        # Conv Layers
        self.p_x_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=(3), stride=(1), padding=(0)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=(3), stride=(1), padding=(1)),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, kernel_size=(2, 5), stride=(1), padding=(1)),
        )

    def encode(self, x):
        # Conv
        out = self.q_z_conv(x)
        self.conv_op_shape = out.shape
        self.q_z_output_dim = out.shape[1] * out.shape[2] * out.shape[3]

        # Flatten
        out = self.flatten(out)
        # Dense
        out = self.q_z_lin(out)
        return out

    def decode(self, z):
        # Dense
        out = self.p_x_lin(z)
        # Unflatten
        out = out.view(
            self.conv_op_shape[0],
            self.conv_op_shape[1],
            self.conv_op_shape[2],
            self.conv_op_shape[3],
        )
        # Conv transpose
        out = self.p_x_conv(out)
        return out

    def forward(self, x):
        z = self.encode(x)
        out = self.decode(z)
        return out

    def get_final_layer_dims(self):
        return self.conv_op_shape

    def set_final_layer_dims(self, conv_op_shape):
        self.conv_op_shape = conv_op_shape


class FPGA_prototype_model(nn.Module):
    def __init__(self, n_features, z_dim, *args, **kwargs):
        super(FPGA_prototype_model, self).__init__(*args, **kwargs)

        # encoder
        self.en1 = nn.Linear(n_features, 20, dtype=torch.float64)
        self.en_act1 = nn.ReLU()
        self.en2 = nn.Linear(20, 10, dtype=torch.float64)
        self.en_act2 = nn.ReLU()
        self.en3 = nn.Linear(10, z_dim, dtype=torch.float64)

        # decoder
        self.de1 = nn.Linear(z_dim, 10, dtype=torch.float64)
        self.de_act1 = nn.ReLU()
        self.de2 = nn.Linear(10, 20, dtype=torch.float64)
        self.de_act2 = nn.ReLU()
        self.de3 = nn.Linear(20, n_features, dtype=torch.float64)

        self.n_features = n_features
        self.z_dim = z_dim

    def encode(self, x):
        s1 = self.en1(x)
        s2 = self.en_act1(s1)
        s3 = self.en2(s2)
        s4 = self.en_act2(s3)
        s5 = self.en3(s4)
        return s5

    def decode(self, z):
        s6 = self.de1(z)
        s7 = self.de_act1(s6)
        s8 = self.de2(s7)
        s9 = self.de_act2(s8)
        s10 = self.de3(s9)
        return s10

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

    def get_layers(self) -> list:
        return [
            self.en1,
            self.en_act1,
            self.en2,
            self.en_act2,
            self.en3,
            self.de1,
            self.de_act1,
            self.de2,
            self.de_act2,
            self.de3,
        ]


class Conv_AE_3D(nn.Module):
    def __init__(self, n_features, z_dim, *args, **kwargs):
        super(Conv_AE_3D, self).__init__(*args, **kwargs)

        self.q_z_mid_dim = 2000
        self.q_z_output_dim = 612864
        self.compress_to_latent_space = True
        self.debug = False

        # Encoder

        # Conv Layers
        self.q_z_conv = nn.Sequential(
            nn.Conv3d(1, 4, kernel_size=(1, 3, 5), stride=(1), padding=(0)),
            # nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv3d(4, 8, kernel_size=(3), stride=(1), padding=(0)),
            # nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv3d(8, 16, kernel_size=(3), stride=(1), padding=(0)),
            # nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv3d(16, 32, kernel_size=(3), stride=(1), padding=(0)),
            # nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=(3), stride=(1), padding=(0)),
            # nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=(1, 3, 3), stride=(1), padding=(0)),
            # nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        # Flatten
        self.flatten = nn.Flatten(start_dim=1)

        # Linear layers
        self.q_z_lin = nn.Sequential(
            nn.Linear(self.q_z_output_dim, self.q_z_mid_dim),
            nn.ReLU(),
            # nn.BatchNorm1d(self.q_z_mid_dim),
            nn.Linear(self.q_z_mid_dim, z_dim),
            nn.ReLU(),
        )

        # Decoder

        # Linear layers
        self.p_x_lin = nn.Sequential(
            nn.Linear(z_dim, self.q_z_mid_dim),
            nn.ReLU(),
            # nn.BatchNorm1d(self.q_z_mid_dim),
            nn.Linear(self.q_z_mid_dim, self.q_z_output_dim),
            nn.ReLU(),
        )

        # Conv Layers
        self.p_x_conv = nn.Sequential(
            nn.ConvTranspose3d(64, 64, kernel_size=(1, 3, 3), stride=(1), padding=(0)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.ConvTranspose3d(64, 32, kernel_size=(3), stride=(1), padding=(0)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.ConvTranspose3d(32, 16, kernel_size=(3), stride=(1), padding=(0)),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.ConvTranspose3d(16, 8, kernel_size=(3), stride=(1), padding=(0)),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.ConvTranspose3d(8, 4, kernel_size=(3), stride=(1), padding=(0)),
            nn.BatchNorm3d(4),
            nn.ReLU(),
            nn.ConvTranspose3d(4, 1, kernel_size=(1, 3, 5), stride=(1), padding=(0)),
        )

    def encode(self, x):
        # Conv

        out = self.q_z_conv(x)
        # Flatten
        out = self.flatten(out)

        if self.compress_to_latent_space:
            # Dense
            out = self.q_z_lin(out)

        return out

    def decode(self, out):
        if self.compress_to_latent_space:
            out = self.p_x_lin(out)

        out = out.view(4, 64, 7, 38, 36)

        # Conv transpose
        out = self.p_x_conv(out)

        return out

    def forward(self, x):
        z = self.encode(x)
        out = self.decode(z)
        return out

    def set_compress_to_latent_space(self, compress_to_latent_space):
        self.compress_to_latent_space = compress_to_latent_space


class Conv_AE_GDN(nn.Module):
    def __init__(self, n_features, z_dim, *args, **kwargs):
        super(Conv_AE_GDN, self).__init__(*args, **kwargs)

        self.q_z_mid_dim = 2000
        self.q_z_output_dim = 128
        self.conv_op_shape = None

        # Encoder

        # Conv Layers
        self.q_z_conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(2, 5), stride=(1), padding=(1)),
            # nn.BatchNorm2d(8),
            GDN(8),
            nn.Conv2d(8, 16, kernel_size=(3), stride=(1), padding=(1)),
            nn.BatchNorm2d(16),
            GDN(16),
            nn.Conv2d(16, 32, kernel_size=(3), stride=(1), padding=(0)),
            # nn.BatchNorm2d(32),
            GDN(32),
        )
        # Flatten
        self.flatten = nn.Flatten(start_dim=1)

        # Linear layers
        self.q_z_lin = nn.Sequential(
            nn.Linear(self.q_z_output_dim, self.q_z_mid_dim),
            nn.ReLU(),
            # nn.BatchNorm1d(self.q_z_output_dim),
            nn.Linear(self.q_z_mid_dim, z_dim),
            nn.ReLU(),
        )

        # Decoder

        # Linear layers
        self.p_x_lin = nn.Sequential(
            nn.Linear(z_dim, self.q_z_mid_dim),
            nn.ReLU(),
            # nn.BatchNorm1d(self.q_z_output_dim),
            nn.Linear(self.q_z_mid_dim, self.q_z_output_dim),
            nn.ReLU()
            # nn.BatchNorm1d(42720)
        )
        # Conv Layers
        self.p_x_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=(3), stride=(1), padding=(0)),
            # nn.BatchNorm2d(16),
            GDN(16, inverse=True),
            nn.ConvTranspose2d(16, 8, kernel_size=(3), stride=(1), padding=(1)),
            # nn.BatchNorm2d(8),
            GDN(8, inverse=True),
            nn.ConvTranspose2d(8, 1, kernel_size=(2, 5), stride=(1), padding=(1)),
        )

    def encode(self, x):
        # Conv
        out = self.q_z_conv(x)
        self.conv_op_shape = out.shape
        self.q_z_output_dim = out.shape[1] * out.shape[2] * out.shape[3]

        # Flatten
        out = self.flatten(out)
        # Dense
        out = self.q_z_lin(out)
        return out

    def decode(self, z):
        # Dense
        out = self.p_x_lin(z)
        # Unflatten
        out = out.view(
            self.conv_op_shape[0],
            self.conv_op_shape[1],
            self.conv_op_shape[2],
            self.conv_op_shape[3],
        )
        # Conv transpose
        out = self.p_x_conv(out)
        return out

    def forward(self, x):
        z = self.encode(x)
        out = self.decode(z)
        return out

    def get_final_layer_dims(self):
        return self.conv_op_shape

    def set_final_layer_dims(self, conv_op_shape):
        self.conv_op_shape = conv_op_shape


class PJ_Conv_AE(nn.Module):
    def __init__(self, n_features, z_dim=10, *args, **kwargs):
        super(PJ_Conv_AE, self).__init__(*args, **kwargs)

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(
                1, 20, kernel_size=5, stride=2, padding=2
            ),  # Adjust input channels to 1 for grayscale images
            nn.LeakyReLU(0.2),
            nn.Conv2d(20, 50, kernel_size=5, stride=2, padding=2),
            nn.Flatten(),
            nn.Linear(50 * 7 * 7, 500),  # Adjust input size based on your data
            nn.Linear(500, z_dim),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 500),
            nn.LeakyReLU(0.2),
            nn.Linear(500, 2450),
            nn.Unflatten(1, (50, 7, 7)),
            nn.ConvTranspose2d(
                50, 20, kernel_size=5, stride=2, padding=2, output_padding=1
            ),
            nn.ConvTranspose2d(
                20, 1, kernel_size=5, stride=2, padding=2, output_padding=1
            ),
            nn.LeakyReLU(0.2),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        out = self.decode(z)
        return out

    def get_final_layer_dims(self):
        return list(self.decoder.children())[-1]

    def set_final_layer_dims(self, conv_op_shape):
        self.conv_op_shape = conv_op_shape


class TransformerAE(nn.Module):
    """Autoencoder mixed with the Transformer Encoder layer

    Args:
        nn (_type_): _description_
    """

    def __init__(
        self,
        in_dim,
        h_dim=256,
        n_heads=1,
        latent_size=50,
        activation=torch.nn.functional.gelu,
    ):
        super(TransformerAE, self).__init__()

        self.transformer_encoder_layer_1 = torch.nn.TransformerEncoderLayer(
            batch_first=True,
            norm_first=True,
            d_model=in_dim,
            activation=activation,
            dim_feedforward=h_dim,
            nhead=n_heads,
        )

        self.transformer_encoder_layer_2 = torch.nn.TransformerEncoderLayer(
            batch_first=True,
            norm_first=True,
            d_model=256,
            activation=activation,
            dim_feedforward=256,
            nhead=n_heads,
        )
        self.transformer_encoder_layer_3 = torch.nn.TransformerEncoderLayer(
            batch_first=True,
            norm_first=True,
            d_model=128,
            activation=activation,
            dim_feedforward=128,
            nhead=n_heads,
        )

        self.encoder_layer_1 = torch.nn.Sequential(
            torch.nn.LazyBatchNorm1d(),
            torch.nn.Linear(in_dim, 256),
            torch.nn.LeakyReLU(),
        )

        self.encoder_layer_2 = torch.nn.Sequential(
            torch.nn.LazyBatchNorm1d(),
            torch.nn.Linear(256, 128),
            torch.nn.LeakyReLU(),
        )

        self.encoder_layer_3 = torch.nn.Sequential(
            torch.nn.LazyBatchNorm1d(),
            torch.nn.Linear(128, latent_size),
            torch.nn.LeakyReLU(),
        )

        self.decoder_layer_3 = torch.nn.Sequential(
            torch.nn.LazyBatchNorm1d(),
            torch.nn.Linear(latent_size, 128),
            torch.nn.LeakyReLU(),
        )
        self.decoder_layer_2 = torch.nn.Sequential(
            torch.nn.LazyBatchNorm1d(), torch.nn.Linear(128, 256), torch.nn.LeakyReLU()
        )
        self.decoder_layer_1 = torch.nn.Sequential(
            torch.nn.LazyBatchNorm1d(),
            torch.nn.Linear(256, in_dim),
            torch.nn.LeakyReLU(),
        )

        self.transformer_decoder_layer_3 = torch.nn.TransformerEncoderLayer(
            batch_first=True,
            d_model=128,
            activation=activation,
            dim_feedforward=128,
            nhead=n_heads,
        )

        self.transformer_decoder_layer_2 = torch.nn.TransformerEncoderLayer(
            batch_first=True,
            d_model=256,
            activation=activation,
            dim_feedforward=256,
            nhead=n_heads,
        )

        self.transformer_decoder_layer_1 = torch.nn.TransformerEncoderLayer(
            d_model=in_dim,
            dim_feedforward=h_dim,
            activation=activation,
            nhead=n_heads,
        )

    def encoder(self, x: torch.Tensor):
        """_summary_

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        z = self.transformer_encoder_layer_1(x)
        z = self.encoder_layer_1(z)
        z = self.transformer_encoder_layer_2(z)
        z = self.encoder_layer_2(z)
        z = self.transformer_encoder_layer_3(z)
        z = self.encoder_layer_3(z)

        return z

    def decoder(self, z: torch.Tensor):
        """_summary_

        Args:
            z (_type_): _description_

        Returns:
            _type_: _description_
        """
        x = self.decoder_layer_3(z)
        x = self.transformer_decoder_layer_3(x)
        x = self.decoder_layer_2(x)
        x = self.transformer_decoder_layer_2(x)
        x = self.decoder_layer_1(x)
        x = self.transformer_decoder_layer_1(x)
        return x

    def forward(self, x: torch.Tensor):
        """_summary_

        Args:
            z (_type_): _description_

        Returns:
            _type_: _description_
        """
        z = self.encoder(x)
        x = self.decoder(z)
        return x
