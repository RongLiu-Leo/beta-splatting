# Deformable Beta Splatting

[![button](https://img.shields.io/badge/Project-Website-blue.svg?style=social&logo=Google-Chrome)](https://rongliu-leo.github.io/beta-splatting/)
[![button](https://img.shields.io/badge/Paper-arXiv-red.svg?style=social&logo=arXiv)](https://arxiv.org/abs/2501.18630)

<span class="author-block">
  <a href="https://rongliu-leo.github.io/">Rong Liu*</a>,
</span>
<span class="author-block">
  <a href=""> Dylan Sun*</a>,
</span>
<span class="author-block">
  <a href="https://www.linkedin.com/in/meida-chen-938a265b/"> Meida Chen</a>,
</span>
<span class="author-block">
  <a href="https://yuewang.xyz/"> Yue Wang†</a>,
</span>
<span class="author-block">
  <a href="https://scholar.google.com/citations?user=JKWxGfsAAAAJ&hl=en"> Andrew Feng†</a>
</span>

(*Co-first authors, equal technical contribution; †Co-advisors, equal leading contribution.)

![Teaser image](assets/teaser.png)



## How to Install

This project is built on top of the [Original 3DGS](https://github.com/graphdeco-inria/gaussian-splatting), [3DGS-MCMC](https://github.com/ubc-vision/3dgs-mcmc) and [gsplat](https://github.com/nerfstudio-project/gsplat) code bases.

### Installation Steps

1. **Clone the Repository:**
   ```sh
   git clone https://github.com/RongLiu-Leo/beta-splatting.git
   cd beta-splatting
   ```
1. **Set Up the Conda Environment:**
    ```sh
    conda create -y -n beta_splatting python=3.8
    conda activate beta_splatting
    ```
1. **Install [Pytorch](https://pytorch.org/get-started/locally/)**
    ```sh
    # Based on your CUDA version
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```
1. **Install Dependencies and Submodules:**
    ```sh
    pip install -r requirements.txt
    cd submodule
    pip install .
    cd ..
    ```

## Citation
If you find our code or paper helps, please consider giving us a star or citing:
```bibtex
@misc{liu2025deformablebetasplatting,
      title={Deformable Beta Splatting}, 
      author={Rong Liu and Dylan Sun and Meida Chen and Yue Wang and Andrew Feng},
      year={2025},
      eprint={2501.18630},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2501.18630}, 
}
```