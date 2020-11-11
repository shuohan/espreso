#!/usr/bin/env python

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
from scipy.ndimage import zoom
from torch.nn.functional import interpolate


scale = 1 / 4
# data = np.array(Image.open('lena.png').convert('L')).astype(np.float32) / 255
data = np.zeros((256, 256), dtype=np.float32)
for i in range(256):
    for j in range(256):
        if np.sqrt((i - 128) ** 2 + (j - 128) ** 2) < 64:
            data[i, j] = 1

data_torch = torch.tensor(data)[None, None, ...]
# scipy_interp2d_data = interp2d
scipy_zoom_filter = zoom(data, scale, order=3, prefilter=True, mode='nearest')
scipy_zoom_nofilter = zoom(data, scale, order=3, prefilter=False, mode='nearest')
torch_interp = interpolate(data_torch, size=scipy_zoom_filter.shape,
                           mode='bicubic', align_corners=True)
torch_interp = torch_interp.detach().numpy().squeeze()
print(scipy_zoom_filter.shape, torch_interp.shape)

print('no filter diff', np.sum(np.abs(scipy_zoom_nofilter - torch_interp)))
print('filter diff', np.sum(np.abs(scipy_zoom_filter - torch_interp)))

plt.subplot(2, 3, 1)
plt.imshow(scipy_zoom_nofilter, cmap='gray')
plt.title('scipy zoom no filter')
plt.subplot(2, 3, 2)
plt.imshow(scipy_zoom_filter, cmap='gray')
plt.title('scipy zoom filter')
plt.subplot(2, 3, 3)
plt.imshow(torch_interp, cmap='gray')
plt.title('torch interp')
plt.subplot(2, 3, 5)
plt.imshow(np.abs(scipy_zoom_filter - scipy_zoom_nofilter), cmap='gray')
plt.subplot(2, 3, 6)
plt.imshow(np.abs(torch_interp - scipy_zoom_nofilter), cmap='gray')
# plt.gcf().savefig('compare_interp.png')

plt.figure()
plt.plot(scipy_zoom_filter[:, 32])
plt.plot(scipy_zoom_nofilter[:, 32])
plt.plot(torch_interp[:, 32])
plt.legend(['filter', 'nofilter', 'torch'])
plt.show()
