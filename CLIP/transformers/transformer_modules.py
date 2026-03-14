import torch 
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, img_size = 224, in_channels = 3, patch_size =32, embed_dim = 512, p_dropout = 0.1):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size) 
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim)) # (1, 1, D)
        num_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(p_dropout)
    
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.patch_embed(x) # (B, D, H/patch_size, W/patch_size)
        x = x.flatten(2).transpose(1, 2) # (B, P, D) where P = num_patches
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1) #  (B, P+1, D)
        x = x + self.pos_embed
        x = self.dropout(x)
        return x

class AttentionHead(nn.Module):
    def __init__(self, head_size, embed_dim=512, p_dropout=0.1):
        super().__init__()
        self.head_size = head_size
        self.query = nn.Linear(embed_dim, head_size)
        self.key = nn.Linear(embed_dim, head_size)
        self.value = nn.Linear(embed_dim, head_size)
        self.dropout = nn.Dropout(p_dropout)
    
    def forward(self, x:torch.Tensor, mask:torch.Tensor = None):
        B, T, D = x.shape
        q = self.query(x) # (B, T, head_size)
        k = self.key(x)   # (B, T, head_size)
        v = self.value(x) # (B, T, head_size)

        attn = q @ k.transpose(1, 2) * (self.head_size ** -0.5) # (B, T, T)
        if mask is not None:
            pad_mask = mask.to(torch.bool).unsqueeze(1) # (B, 1, T)
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(attn, dim=-1) # (B, T, T)
        attn = self.dropout(attn)
        out = attn @ v # (B, T, head_size)
        return out

def main():
    B = 10
    T = 7
    img_size = 224
    in_channels = 3
    D = 512

    input = torch.zeros(B, in_channels, img_size, img_size) # (B, C, H, W)
    model = PatchEmbedding()
    output = model(input)
    print(output.shape) # (B, P+1, D)

    input = torch.zeros(B, T, D) # (B, N, D)
    model = AttentionHead(head_size=8, embed_dim=D, p_dropout=0.1)
    output = model(input)
    print(output.shape) # (B, N, head_size)

if __name__ == "__main__":
    main()