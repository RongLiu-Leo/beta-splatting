# Beta Splatting

[![button](https://img.shields.io/badge/Project%20Website-orange?style=for-the-badge)](https://ubc-vision.github.io/3dgs-mcmc/)
[![button](https://img.shields.io/badge/Paper-blue?style=for-the-badge)](https://arxiv.org/abs/2404.09591)

<!-- <span class="author-block">
  <a href="https://shakibakh.github.io/">Shakiba Kheradmand</a>,
</span>
<span class="author-block">
  <a href="http://drebain.com/"> Daniel Rebain</a>,
</span>
<span class="author-block">
  <a href="https://hippogriff.github.io/"> Gopal Sharma</a>,
</span>
<span class="author-block">
  <a href="https://wsunid.github.io/"> Weiwei Sun</a>,
</span>
<span class="author-block">
  <a href="https://scholar.google.com/citations?user=1iJfq7YAAAAJ&hl=en"> Yang-Che Tseng</a>,
</span>
<span class="author-block">
  <a href="http://www.hossamisack.com/">Hossam Isack</a>,
</span>
<span class="author-block">
  <a href="https://abhishekkar.info/">Abhishek Kar</a>,
</span>
<span class="author-block">
  <a href="https://taiya.github.io/">Andrea Tagliasacchi</a>,
</span>
<span class="author-block">
  <a href="https://www.cs.ubc.ca/~kmyi/">Kwang Moo Yi</a>
</span>

<hr> -->

<!-- <video controls>
  <source src="docs/resources/training_rand_compare/bicycle_both-rand.mp4" type="video/mp4">
</video> -->

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
</section> -->



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
    pip install submodules\simple-knn
    pip install submodules\diff-gaussian-rasterization
    ```

