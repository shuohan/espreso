#!/usr/bin/env python

import torch


class PermConv(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(64, 256, 3, bias=False)

    def forward(self, x):
        nb, nc, nx, ny = x.shape
        output = x.permute(0, 2, 1, 3)
        output = output.reshape(nb * nx, nc, ny)
        output = self.conv(output)
        output = output.view(nb, nx, 256, ny - 2)
        output = output.permute(0, 2, 1, 3)
        return output


x = torch.rand(10, 64, 128, 128).cuda()
# nb, nc, nx, ny = x.shape
# x = x.permute(0, 2, 1, 3)
# x = x.reshape(nb * nx, nc, ny)

torch.manual_seed(0)
conv2 = torch.nn.Conv2d(64, 256, (3, 1), bias=False).cuda()

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
out2 = conv2(x)
y = torch.sum(out2)
y.backward()
end.record()

torch.cuda.synchronize()
print(start.elapsed_time(end))

torch.manual_seed(0)
conv1 = PermConv().cuda()

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
out1 = conv1(x)
y = torch.sum(out1)
y.backward()
end.record()

torch.cuda.synchronize()
print(start.elapsed_time(end))

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
out2 = conv2(x)
y = torch.sum(out2)
y.backward()
end.record()

torch.cuda.synchronize()
print(start.elapsed_time(end))
