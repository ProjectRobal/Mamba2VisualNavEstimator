import torch
import torch.nn as nn


# This is inspired by Kolmogorov-Arnold Networks but using Chebyshev polynomials instead of splines coefficients
class ChebyKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree):
        super(ChebyKANLayer, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.degree = degree
        self.norm = nn.RMSNorm(input_dim)

        self.cheby_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.normal_(self.cheby_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))
        self.register_buffer("arange", torch.arange(0, degree + 1, 1))

    def forward(self, x):
        shape_len = len(x.shape)
        
        if shape_len == 2:
            batch_size = x.shape[0]
            x = x.reshape((batch_size, self.inputdim))
        elif shape_len == 3:
            batch_size, seq_len = x.shape[0], x.shape[1]
            x = x.reshape((batch_size * seq_len, self.inputdim))
        # Since Chebyshev polynomial is defined in [-1, 1]
        # We need to normalize x to [-1, 1] using tanh
        # x = torch.tanh(x)
        x = self.norm(x)
        # x = torch.tanh(x)
        # View and repeat input degree + 1 times
        x = x.view((-1, self.inputdim, 1)).expand(
            -1, -1, self.degree + 1
        )  # shape = (batch_size, inputdim, self.degree + 1)
        # Apply acos
        x = x.acos()
        # Multiply by arange [0 .. degree]
        x *= self.arange
        # Apply cos
        x = x.cos()
        # Compute the Chebyshev interpolation
        y = torch.einsum(
            "bid,iod->bo", x, self.cheby_coeffs
        )  # shape = (batch_size, outdim)
        
        if shape_len == 2:
            y = y.reshape((-1, self.outdim))
        elif shape_len == 3:
            y = y.reshape((batch_size, seq_len, self.outdim))
            
        return y