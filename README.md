# Structured Noise Generation

[Project Page](https://yuzeng-at-tri.github.io/ppd-page/)


## For ComfyUI
clone this repo into custom_nodes

## For CLI

first install this repo

```bash
pip install .
```

Or for development mode:

```bash
pip install -e .
```

Usage: 

```
python -m structured_noise.structured_noise_pytorch --path_in dog.jpg --path_out dog_structured_noise.png
```

refer to https://github.com/zengxianyu/PPD-examples for usage in training or inference


If you find this work useful, please cite:

```
@article{zeng2025neuralremaster,
  title   = {{NeuralRemaster}: Phase-Preserving Diffusion for Structure-Aligned Generation},
  author  = {Zeng, Yu and Ochoa, Charles and Zhou, Mingyuan and Patel, Vishal M and
             Guizilini, Vitor and McAllister, Rowan},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2025}
}
```
