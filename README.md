# Visualization Anythings 
> it helps those who want to study Transformer.. <br/>
> [Wongi Park](https://www.linkedin.com/in/wongipark/) <br/>

## Environment Setting
- **Conda environment**
: Ubuntu 18.04 CUDA-10.1 (10.2) with Pytorch==1.13.0, Torchvision==0.6.0 (python 3.8~11), timm, reception Fields<br/>
```
# Create Environment
conda create -n vslt python=3.8
conda activate vslt

# Install pytorch, torchvision, cudatoolkit
conda install pytorch==1.13.0 torchvision==0.6.0 cudatoolkit=10.1 (10.2) -c pytorch
conda install timm
conda install receptivefield>=0.5.0
```

![](pngs/visual.png)

***Q1. What properties of visualize about attention or amplitude in papers?***  

A1. These visualizations aid in interpreting the model's behavior, analyzing its strengths and weaknesses, and guiding further improvements and research in the field.


***Q2. Why we visualize Tokens interactions?***  

A2. This provides insights into how attention is propagated and shared across the input patches. Analyzing token interactions can help understand the flow of information and the dependencies learned by the self-attention mechanism.


## Supporting papers

> (1) DINOv2  ([Paper](https://arxiv.org/abs/2304.07193) / [Code](https://github.com/facebookresearch/dinov2)) <br/>
> (2) How Do Vision Transformers Work? ([Paper](https://arxiv.org/abs/2202.06709) / [Code](https://github.com/xxxnell/how-do-vits-work/tree/transformer)) <br/>
> (3) More ConvNets in the 2020s: Scaling up Kernels Beyond 51x51 using Sparsity ([Paper](https://arxiv.org/abs/2207.03620) / [Code](https://github.com/VITA-Group/SLaK)) <br/>
> (4) An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale ([Paper](https://arxiv.org/abs/2010.11929) / [Code](https://github.com/lucidrains/vit-pytorch)) <br/>


<!-- ## Su
### 1. [DINOv2](https://arxiv.org/abs/2304.07193)
### 2. [PCA Visualized](https://github.com/purnasai/Dino_V2)
### 3. [Attention Distance Visualized](https://github.com/all-things-vits/code-samples)
### 4. [Amplitude GradCAM](https://github.com/all-things-vits/code-samples)
### 5. [ReceptionFields GradCAM](https://github.com/shelfwise/receptivefield/blob/master/notebooks/minimal_example_with_pytorch_API.ipynb)
### 6. [Effective Receptive Field](hhttps://github.com/DingXiaoH/RepLKNet-pytorch/blob/main/erf/visualize_erf.py)
### 7. [Fourier GradCAM](https://github.com/xxxnell/how-do-vits-work/blob/transformer/fourier_analysis.ipynb)
### 8. [Masked Visualized](https://github.com/youweiliang/evit/blob/master/visualize_mask.py) -->

## How to cite
```
@article = {
    title = {Visualize Anything},
    author = {Wongi Park},
    journal = {GitHub},
    url = {https://github.com/kalelpark/visualizing},
    year = {2023},
}
```