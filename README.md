
# Full code release soon :)!
# `SceneTok`: A Compressed, Diffusable Token Space for 3D Scenes [CVPR 2026] 
<a href="https://mohammadasim98.github.io">Mohammad Asim</a>, <a href="https://geometric-rl.mpi-inf.mpg.de/people/Wewer.html">Christopher Wewer</a>, <a href="https://geometric-rl.mpi-inf.mpg.de/people/lenssen.html">Jan Eric Lenssen</a>

*Max Planck Institute for Informatics, Saarland Informatics Campus*

<h4 align="left">
<a href="https://geometric-rl.mpi-inf.mpg.de/scenetok/">Project Page</a>
</h4>

### `TL;DR: Proposes a scene autoencoder to compress the view-set of a scene into a compressed, unstructured 1D token representation.` 

### 📣 News

- **23.02.2026** - Initial repository releases.
- **21.02.2026** - Accepted to [`CVPR 2026`](https://cvpr.thecvf.com/) 🎉!

## 🔍 Method Overview 
<div align="center">
  <img src="assets/imgs/scenetok_method.svg" width="800"/>
  
</div>
<div align="justify">
<p>
  
  <b>(Left)</b> `SceneTok` encodes view sets into a set of compressed, unstructured scene tokens by chaining a VA-VAE image compressor and a perceiver module. The tokens can be rendered from novel views with a generative decoder based on rectified flows. 
  
  <b>(Right)</b> `SceneGen` perform scene generation by generating compressed scene tokens conditioned on a single or a few images and a set of anchor poses, defining the spatial scene extent.
</p>
</div>


## ✅ TODO
`SceneTok` Version
- [ ] RealEstate10K
- [ ] DL3DV

`SceneGen` version
- [ ] RealEstate10K


## BibTeX
If you are planning to use `SceneTok` in your work, consider citing it as follows.
<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <pre><code>@inproceedings{asim26scenetok,
    title = {SceneTok: A Compressed, Diffusable Token Space for 3D Scenes},
    author = {Asim, Mohammad and Wewer, Christopher and Lenssen, Jan Eric},
    booktitle = {Computer Vision and Pattern Recognition ({CVPR})},
    year = {2026},
}</code></pre>
  </div>
</section>

