import torch
import torch.nn as nn
import math

class Tf(nn.Module):
    """
    input: Tensor -> [B, C, H, W, D]
    return: Tensor -> [B, C, H, W, D]
    """
    def __init__(self, in_channels, k=8, P=2):
        # k: int: num_heads
        super().__init__()

        if (math.pow(P, 3) * in_channels) % k != 0:
            raise ValueError("hidden size should be divisible by k.")

        self.P = P
        self.conv_k = nn.Conv3d(in_channels, in_channels, 1, stride=1, padding=0)
        self.conv_q = nn.Conv3d(in_channels, in_channels, 1, stride=1, padding=0)
        self.conv_v = nn.Conv3d(in_channels, in_channels, 1, stride=1, padding=0)

        self.dk = int(math.pow(P, 3) * in_channels / k)
        self.k = k

        self.norm1 = nn.LayerNorm(in_channels)

        self.stem = nn.Conv3d(in_channels, in_channels, 3, stride=1, padding=1)

        self.conv = nn.Conv3d(in_channels, in_channels, 3, stride=1, padding=1)
        self.norm2 = nn.LayerNorm(in_channels)

    def forward(self, x):
        B, C, H, W, D = x.shape
        q, k, v = self.conv_k(x), self.conv_q(x), self.conv_v(x)

        N = int(H * W * D / math.pow(self.P, 3))
        q = q.permute(0, 2, 3, 4, 1).reshape(B, N, -1, C).reshape(B, N, self.k, self.dk).transpose(1, 2).contiguous()
        k = k.permute(0, 2, 3, 4, 1).reshape(B, N, -1, C).reshape(B, N, self.k, self.dk).transpose(1, 2).contiguous()
        v = v.permute(0, 2, 3, 4, 1).reshape(B, N, -1, C).reshape(B, N, self.k, self.dk).transpose(1, 2).contiguous()

        k = torch.transpose(k, 2, 3)
        att_mat = ((q @ k) / (self.dk ** 0.5)).softmax(-1)
        att_mat = att_mat @ v

        x = x.permute(0, 2, 3, 4, 1)
        att_mat = att_mat.transpose(1, 2).reshape(B, N, -1, C).reshape(B, H, W, D, C)
        att = self.norm1(att_mat + x)  # Y

        x = att.permute(0, 4, 1, 2, 3)
        x = self.stem(x)
        x = self.norm2(self.conv(x).permute(0, 2, 3, 4, 1) + att)

        return x.permute(0, 4, 1, 2, 3)

if __name__ == '__main__':
    im = torch.randn(2, 64, 32, 32, 32)#    .mat
    m = Tf(64, 2, 2)
    print(m)
    y = m(im)
    print(y.shape)

'''new'''
# class Tf(nn.Module):
#     """
#     input: Tensor -> [B, C, H, W, D]
#     return: Tensor -> [B, C, H, W, D]
#     """
#     def __init__(self, in_channels, k=8, P=2):
#         # k: int: num_heads
#         super().__init__()
#
#         if (math.pow(P, 3) * in_channels) % k != 0:
#             raise ValueError("hidden size should be divisible by k.")
#
#         self.P = P
#         self.conv_k = nn.Linear(in_channels, in_channels)#nn.Conv3d(in_channels, in_channels, 1, stride=1, padding=0)
#         self.conv_q = nn.Linear(in_channels, in_channels)#nn.Conv3d(in_channels, in_channels, 1, stride=1, padding=0)
#         self.conv_v = nn.Linear(in_channels, in_channels) #nn.Conv3d(in_channels, in_channels, 1, stride=1, padding=0)
#
#         self.dk = int(math.pow(P, 3) * in_channels / k)
#         self.k = k
#
#         self.norm1 = nn.LayerNorm(in_channels)
#
#         self.stem = nn.Linear(in_channels, in_channels)#nn.Conv3d(in_channels, in_channels, 3, stride=1, padding=1)
#
#         self.conv = nn.Linear(in_channels, in_channels)#nn.Conv3d(in_channels, in_channels, 3, stride=1, padding=1)
#         self.norm2 = nn.LayerNorm(in_channels)
#
#     def forward(self, x):
#         B, C, H, W, D = x.shape
#         print(x.shape)
#         x = x.permute(0,2,3,4,1)
#         print(x.shape)# 2 96 96 16 8
#         q, k, v = self.conv_k(x), self.conv_q(x), self.conv_v(x)
#         print(q.shape,k.shape,v.shape)
#
#         N = int(H * W * D / math.pow(self.P, 3))
#         q = q.permute(0, 2, 3, 4, 1).reshape(B, N, -1, C).reshape(B, N, self.k, self.dk).transpose(1, 2).contiguous()
#         k = k.permute(0, 2, 3, 4, 1).reshape(B, N, -1, C).reshape(B, N, self.k, self.dk).transpose(1, 2).contiguous()
#         v = v.permute(0, 2, 3, 4, 1).reshape(B, N, -1, C).reshape(B, N, self.k, self.dk).transpose(1, 2).contiguous()
#         print(q.shape, k.shape, v.shape)
#
#         k = torch.transpose(k, 2, 3)
#         att_mat = ((q @ k) / (self.dk ** 0.5)).softmax(-1)
#         att_mat = att_mat @ v
#
#         # x = x.permute(0, 2, 3, 4, 1)# 2 96 16 8 96
#         att_mat = att_mat.transpose(1, 2).reshape(B, N, -1, C).reshape(B, H, W, D, C)
#         att = self.norm1(att_mat + x)  # Y
#
#         x = att#.permute(0, 4, 1, 2, 3)
#         x = self.stem(x) #mlp(x)
#         # print(self.conv(x).permute(0, 2, 3, 4, 1).shape)
#         x = self.norm2(self.conv(x).permute(0, 2, 1, 3, 4) + att)
#
#         return x.permute(0, 4, 1, 2, 3)
#
# if __name__ == '__main__':
#     im = torch.randn(2, 8, 96, 96, 16)
#     m = Tf(8, 2, 4)
#     print(m)
#     y = m(im)
#     print(y.shape)
