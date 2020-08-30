import torch
from torch import nn


class MaskL2Loss(nn.Module):
    def __init__(self):
        super(MaskL2Loss, self).__init__()

    def forward(self, pred, target, mask, detach=True, reduce_batch=True):
        assert mask.max() <= 1 + 1e-6
        if detach:
            target = target.detach()
        mask = mask.detach()
        if mask.shape[1] > 1:
            mask = mask[:, 0, :, :].unsqueeze(1)
        assert pred.shape == target.shape
        dif = (pred - target) ** 2 * (mask.float())
        loss = torch.sum(dif.reshape(mask.shape[0], -1).contiguous(), 1)
        count = torch.sum(mask.reshape(mask.shape[0], -1).contiguous(), 1).detach()
        loss[count == 0] = loss[count == 0] * 0
        loss = loss / (count + 1)
        if reduce_batch:
            non_zero_count = torch.sum((count > 0).float())
            if non_zero_count == 0:
                loss = torch.sum(loss) * 0
            else:
                loss = torch.sum(loss) / non_zero_count
            return loss
        else:
            return loss


if __name__ == "__main__":
    l = MaskL2Loss().cuda()
    # l2 = MaskMSELoss().cuda()
    x = torch.rand(2, 3, 128, 128).cuda()
    y = torch.rand(2, 3, 128, 128).cuda()
    m = torch.cat((torch.zeros(1, 3, 128, 128), torch.zeros(1, 3, 128, 128)), 0).cuda()
    m = (m > 0.5).float()
    loss = l(x, y, m)
    # loss2 = l2(x, y, m)
