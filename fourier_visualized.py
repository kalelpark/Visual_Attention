from torchvision.transforms import transforms
from PIL import Image
import copy
import timm
import numpy as np
import torch
import torch.nn as nn
import math

import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from PIL import Image
from einops import rearrange, reduce, repeat


name = "vit_tiny_patch16_224"
model = timm.create_model(name, pretrained=True)

class PatchEmbed(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = copy.deepcopy(model)
        
    def forward(self, x, **kwargs):
        x = self.model.patch_embed(x)
        cls_token = self.model.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.model.pos_drop(x + self.model.pos_embed)
        return x
    
class Residual(nn.Module):
    def __init__(self, *fn):
        super().__init__()
        self.fn = nn.Sequential(*fn)
        
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x
    
    
class Lambda(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        
    def forward(self, x):
        return self.fn(x)


def flatten(xs_list):
    return [x for xs in xs_list for x in xs]

blocks = [
    PatchEmbed(model),
    *flatten([[Residual(b.norm1, b.attn), Residual(b.norm2, b.mlp)] 
              for b in model.blocks]),
    nn.Sequential(model.norm, Lambda(lambda x: x[:, 0]), model.head),
]

normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])

trans_aug = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize,
])

imgs = Image.open("pngs/dog.png")
xs = trans_aug(imgs).unsqueeze(0)

latents = []
with torch.no_grad():
    for block in blocks:
        xs = block(xs)
        latents.append(xs)

latents = [latent[:,1:] for latent in latents]
latents = latents[:-1]  # drop logit (output)

def fourier(x):  # 2D Fourier transform
    f = torch.fft.fft2(x)
    f = f.abs() + 1e-6
    f = f.log()
    return f


def shift(x):  # shift Fourier transformed feature map
    b, c, h, w = x.shape
    return torch.roll(x, shifts=(int(h/2), int(w/2)), dims=(2, 3))

def make_segments(x, y):  # make segment for `plot_segment`
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments

def plot_segment(ax, xs, ys, cmap_name="plasma"):  # plot with cmap segments
    z = np.linspace(0.0, 1.0, len(ys))
    z = np.asarray(z)
    
    cmap = cm.get_cmap(cmap_name)
    norm = plt.Normalize(0.0, 1.0)
    segments = make_segments(xs, ys)
    lc = LineCollection(segments, array=z, cmap=cmap_name, norm=norm,
                        linewidth=2.5, alpha=1.0)
    ax.add_collection(lc)

    colors = [cmap(x) for x in xs]
    ax.scatter(xs, ys, color=colors, marker=marker, zorder=100)

fourier_latents = []
for latent in latents:  # `latents` is a list of hidden feature maps in latent spaces
    latent = latent.cpu()
    
    if len(latent.shape) == 3:  # for ViT
        b, n, c = latent.shape
        h, w = int(math.sqrt(n)), int(math.sqrt(n))
        latent = rearrange(latent, "b (h w) c -> b c h w", h=h, w=w)
    elif len(latent.shape) == 4:  # for CNN
        b, c, h, w = latent.shape
    else:
        raise Exception("shape: %s" % str(latent.shape))
    latent = fourier(latent)
    latent = shift(latent).mean(dim=(0, 1))
    latent = latent.diag()[int(h/2):]  # only use the half-diagonal components
    latent = latent - latent[0]  # visualize 'relative' log amplitudes 
                                 # (i.e., low-freq amp - high freq amp)
    fourier_latents.append(latent)
    
fig, ax1 = plt.subplots(1, 1, figsize=(3.3, 4), dpi=150)
for i, latent in enumerate(reversed(fourier_latents[:-1])):
    freq = np.linspace(0, 1, len(latent))
    ax1.plot(freq, latent, color=cm.plasma_r(i / len(fourier_latents)))
    
ax1.set_xlim(left=0, right=1)

ax1.set_xlabel("Frequency")
ax1.set_ylabel("$\Delta$ Log amplitude")

from matplotlib.ticker import FormatStrFormatter
ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax1.xaxis.set_major_formatter(FormatStrFormatter('%.1fπ'))

if  name == "vit_tiny_patch16_224":  # for ViT-Ti
    pools = []
    msas = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23,]
    marker = "o"

depths = range(len(fourier_latents))

# Normalize
depth = len(depths) - 1
depths = (np.array(depths)) / depth
pools = (np.array(pools)) / depth
msas = (np.array(msas)) / depth

fig, ax2 = plt.subplots(1, 1, figsize=(6.5, 4), dpi=120)
plot_segment(ax2, depths, [latent[-1] for latent in fourier_latents])  # high-frequency component

for pool in pools:
    ax2.axvspan(pool - 1.0 / depth, pool + 0.0 / depth, color="tab:blue", alpha=0.15, lw=0)
for msa in msas:
    ax2.axvspan(msa - 1.0 / depth, msa + 0.0 / depth, color="tab:gray", alpha=0.15, lw=0)
    
ax2.set_xlabel("Normalized depth")
ax2.set_ylabel("$\Delta$ Log amplitude")
ax2.set_xlim(0.0, 1.0)

from matplotlib.ticker import FormatStrFormatter
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
plt.savefig('fourier_amplitude.png')