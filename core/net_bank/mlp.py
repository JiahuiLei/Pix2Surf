"""
2019.8.10
"""
import torch
from torch import nn


class BasicMLPBlock(nn.Module):
    def __init__(self, f_in, p_in, f_out, p_out):
        super().__init__()
        self.f_in = f_in
        self.f_out = f_out
        self.p_in = p_in
        self.p_out = p_out
        self.f = nn.Sequential(
            nn.Conv2d(in_channels=f_in + p_in, out_channels=f_out, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=f_out, out_channels=f_out + p_out, kernel_size=1, stride=1, padding=0),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, f_and_p):
        assert f_and_p.shape[1] == self.f_in + self.p_in
        raw_feature = self.f(f_and_p)
        f = self.relu(raw_feature[:, self.p_out:, :, :])
        xyz = raw_feature[:, :self.p_out, :, :]
        act_feature = torch.cat((xyz, f), 1)
        return act_feature


class NOCS_MLP(nn.Module):
    def __init__(self, c_in=512, p_in=2, p_out=3, bw=None, compress=True):
        super().__init__()
        self.bw = c_in if bw is None else bw
        self.p_out = p_out
        self.block1 = BasicMLPBlock(f_in=c_in, p_in=p_in, f_out=self.bw, p_out=p_out)
        self.block2 = BasicMLPBlock(f_in=self.bw, p_in=p_out, f_out=self.bw, p_out=p_out)
        self.block3v = BasicMLPBlock(f_in=self.bw, p_in=p_out, f_out=self.bw, p_out=p_out)
        self.block4v = BasicMLPBlock(f_in=self.bw, p_in=p_out, f_out=self.bw, p_out=p_out)
        self.layer5v = nn.Conv2d(in_channels=self.bw + p_out, out_channels=p_out, kernel_size=1, stride=1, padding=0)
        self.compress_fn = nn.Sigmoid()
        self.compress = compress

    def forward(self, code, uv, unique_code=True):
        if unique_code:
            input = torch.cat((code.unsqueeze(2).unsqueeze(2).repeat(1, 1, uv.shape[2], uv.shape[3]), uv),
                              1).contiguous()
        else:
            input = torch.cat((code, uv), 1).contiguous()

        f1 = self.block1(input)
        xyz1 = f1[:, :self.p_out, :, :]

        f2 = self.block2(f1)
        xyz2 = f2[:, :self.p_out, :, :]

        f3_v = self.block3v(f2)
        xyz3_v = f3_v[:, :self.p_out, :, :]

        f4_v = self.block4v(f3_v)
        xyz4_v = f4_v[:, :self.p_out, :, :]

        xyz5_v = self.layer5v(f4_v)

        if self.compress:
            xyz_v = self.compress_fn(xyz1 + xyz2 + xyz3_v + xyz4_v + xyz5_v)
        else:
            xyz_v = xyz1 + xyz2 + xyz3_v + xyz4_v + xyz5_v
        return xyz_v

class NOCS_AMP_MLP(nn.Module):
    def __init__(self, latent_dim=1024, amp_dim=256, p_in=2, c_out=3):
        super().__init__()
        self.main_mlp = NOCS_MLP(c_in=latent_dim, p_in=amp_dim, p_out=c_out, bw=512, compress=False)
        self.amp1 = torch.nn.Conv2d(p_in, amp_dim // 4, 1)
        self.amp2 = torch.nn.Conv2d(amp_dim // 4, amp_dim // 2, 1)
        self.amp3 = torch.nn.Conv2d(amp_dim // 2, amp_dim, 1)
        self.activation = nn.ReLU()

    def forward(self, code, uv, unique_code=True):
        am = self.activation(self.amp1(uv))
        am = self.activation(self.amp2(am))
        am = self.activation(self.amp3(am))
        return self.main_mlp(code, am, unique_code)

if __name__ == "__main__":
    uv = torch.rand(5, 2, 100, 1).cuda()
    c = torch.rand(5, 512).cuda()
    net = NOCS_MLP().cuda()
    y = net(c, uv)
    print(y.shape)

    net = NOCS_AMP_MLP(1024, 256, 2, 3).cuda()
    c = torch.rand(3, 1024).cuda()
    uv = torch.rand(3, 2, 100, 1).cuda()
    xyz = net(c, uv, True)
    print(xyz.shape)

