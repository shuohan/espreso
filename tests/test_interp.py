#!/usr/bin/env python

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


# data = torch.arange(20)[None, None, ...].float()
# output = F.interpolate(data, scale_factor=0.5, mode='linear').squeeze()
# print(data)
# print(output)

data = plt.imread('lena.png')[:256, :256, 0]
data = torch.tensor(data, requires_grad=True).float()
fft = torch.rfft(data, 1, onesided=False)
fft = torch.roll(fft, shifts=fft.size(1)//2, dims=1)
fft = F.pad(fft, (0, 0, 50, 50))

new_data = torch.roll(fft, shifts=(fft.size(0)//2, fft.size(0)//2), dims=(0, 1))
new_data = torch.irfft(new_data, 1)

# numpy_fft = np.fft.fftshift(np.fft.fft2(data.detach().numpy()))
# numpy_fft =np.fft.fft2(data.detach().numpy())
# assert np.allclose(np.real(numpy_fft), fft[:, :, 0].detach().numpy())
# diff = np.abs(np.real(numpy_fft) - fft[:, :, 1].detach().numpy())
# diff = diff / np.abs(np.real(numpy_fft))
# print(np.max(diff), np.min(diff))
# plt.imshow(np.abs(np.real(numpy_fft) - fft[:, :, 0].detach().numpy()))

# loss = torch.sum(new_data)
# loss.backward()

# print(new_data.shape)

print(fft.shape)
fft_mag = torch.sqrt(fft[..., 0] ** 2 + fft[..., 1] ** 2)
fft_mag = fft_mag.detach().cpu().numpy().squeeze()

plt.imshow(np.log(fft_mag + 1))
plt.gcf().savefig('fft.png')
# 
# plt.figure()
# plt.subplot(1, 2, 1)
# plt.imshow(data.detach().numpy(), cmap='gray')
# plt.subplot(1, 2, 2)
# plt.imshow(new_data.detach().numpy(), cmap='gray')
# plt.gcf().savefig('fft_image.png')
