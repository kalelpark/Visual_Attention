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

imgs = Image.open("pngs/corgi_image.jpg")
xs = trans_aug(imgs).unsqueeze(0)

latents = []
with torch.no_grad():
    for block in blocks:
        xs = block(xs)
        latents.append(xs)

latents = [latent[:,1:] for latent in latents]
latents = latents[:-1]  # drop logit (output)

# aggregate feature map variances
variances = []
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
                
    variances.append(latent.var(dim=[-1, -2]).mean(dim=[0, 1]))
    

# Plot Fig 9: "Feature map variance"
import numpy as np
import matplotlib.pyplot as plt

pools = []
msas = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23,]
marker = "o"
color = "tab:red"
depths = range(len(variances))

# normalize
depth = len(depths) - 1
depths = (np.array(depths)) / depth
pools = (np.array(pools)) / depth
msas = (np.array(msas)) / depth


fig, ax = plt.subplots(1, 1, figsize=(6.5, 4), dpi=200)
ax.plot(depths, variances, marker=marker, color=color, markersize=7)

for pool in pools:
    ax.axvspan(pool - 1.0 / depth, pool + 0.0 / depth, color="tab:blue", alpha=0.15, lw=0)
for msa in msas:
    ax.axvspan(msa - 1.0 / depth, msa + 0.0 / depth, color="tab:gray", alpha=0.15, lw=0)
    
ax.set_xlim(left=0, right=1.0)
ax.set_ylim(bottom=0.0,)

ax.set_xlabel("Normalized depth")
ax.set_ylabel("Feature map variance")
plt.savefig('fourier_featureMap.png')