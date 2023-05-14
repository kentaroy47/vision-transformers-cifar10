    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        out = torch.matmul(dots, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)