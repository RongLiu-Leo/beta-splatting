# Deformable Beta Splatting

[![button](https://img.shields.io/badge/Project%20Website-orange?style=for-the-badge)](https://rongliu-leo.github.io/beta-splatting/)
[![button](https://img.shields.io/badge/Paper-blue?style=for-the-badge)](https://arxiv.org/abs/2501.18630)

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



<!-- <section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <h2 class="title">BibTeX</h2>
    <pre><code>@article{kheradmand20243d,
  title={3D Gaussian Splatting as Markov Chain Monte Carlo},
  author={Kheradmand, Shakiba and Rebain, Daniel and Sharma, Gopal and Sun, Weiwei and Tseng, Jeff and Isack, Hossam and Kar, Abhishek and Tagliasacchi, Andrea and Yi, Kwang Moo},
  journal={arXiv preprint arXiv:2404.09591},
  year={2024}
}</code></pre>
  </div>
</section>



## How to Install

<!-- This project is built on top of the [Original 3DGS code base](https://github.com/graphdeco-inria/gaussian-splatting) and has been tested only on Ubuntu 20.04. If you encounter any issues, please refer to the [Original 3DGS code base](https://github.com/graphdeco-inria/gaussian-splatting) for installation instructions. -->

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
1. **Install Pytorch**
    ```sh
    # Based on your CUDA version
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```
1. **Install Dependencies and Submodules:**
    ```sh
    pip install plyfile tqdm
    pip install git+https://github.com/rahul-goel/fused-ssim/
    pip install submodules/simple-knn
    pip install submodules/diff-gaussian-rasterization
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