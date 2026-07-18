import torch
import torch.nn as nn

class BidirectionalCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.q1, self.k1, self.v1 = nn.Linear(dim, dim, bias=False), nn.Linear(dim, dim, bias=False), nn.Linear(dim, dim, bias=False)
        self.q2, self.k2, self.v2 = nn.Linear(dim, dim, bias=False), nn.Linear(dim, dim, bias=False), nn.Linear(dim, dim, bias=False)
        
        self.proj1 = nn.Linear(dim, dim)
        self.proj2 = nn.Linear(dim, dim)

    def forward(self, x1, x2):
        B, N1, C = x1.shape
        _, N2, _ = x2.shape

        q1 = self.q1(x1).reshape(B, N1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k2 = self.k2(x2).reshape(B, N2, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v2 = self.v2(x2).reshape(B, N2, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn1 = (q1 @ k2.transpose(-2, -1)) * self.scale
        out1 = (attn1.softmax(dim=-1) @ v2).transpose(1, 2).reshape(B, N1, C)
        out1 = self.proj1(out1)

        q2 = self.q2(x2).reshape(B, N2, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k1 = self.k1(x1).reshape(B, N1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v1 = self.v1(x1).reshape(B, N1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn2 = (q2 @ k1.transpose(-2, -1)) * self.scale
        out2 = (attn2.softmax(dim=-1) @ v1).transpose(1, 2).reshape(B, N2, C)
        out2 = self.proj2(out2)

        return out1, out2
