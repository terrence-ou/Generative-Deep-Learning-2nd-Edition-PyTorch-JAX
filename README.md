# **Generative Deep Learning 2nd Edition in JAX and PyTorch 2.0**

This repository includes the Pytorch 2.0 and JAX implementations of examples in **Generative Deep Learning 2nd Edition** by *David Foster*. You can find the physical/kindle book on [Amazon](https://www.amazon.com/Generative-Deep-Learning-Teaching-Machines/dp/1098134184/ref=sr_1_1?keywords=generative+deep+learning%2C+2nd+edition&qid=1684898042&sprefix=generative%2Caps%2C96&sr=8-1&ufe=app_do%3Aamzn1.fos.006c50ae-5d4c-4777-9bc0-4513d670b6bc) or read it online through [O'REILLY library](https://learning.oreilly.com/home/) (paid subscription needed).

### **Motivation of this project**
I started my journey of deep learning with the help of the first edition of this book. The author introduces topics of generative deep learning in clear and concise way that hels me quickly grasp the key points of type of algorithms without being freak out by heavy mathematics. The codes in the book, written in Tensorflow and Keras, helps me quickly making theories into practice.<br>
We now have other popular deep learning frameworks, like PyTorch and JAX, used by various ML communities. Therefore, I want to translate the Tensorflow and Keras code provided in the book to PyTorch and JAX to help more people study this valuable book more easily.

### **File structure**
The files are organized by frameworks:
```bash
├── JAX_FLAX
│   ├── chapter_**_**
│   │   ├── **.ipynb
│   ├── requirements.txt
├── PyTorch
│   ├── chapter_**_**
│   │   ├── **.ipynb
│   ├── requirements.txt
├── .gitignore
```

## **Environment setup**
I recommend using the separated environments for PyTorch and JAX to avoid potential conflicts on CUDA versions or other related packages. I use [`miniconda`](https://docs.conda.io/en/latest/miniconda.html) to help managing packages for both environments.<br>

Configure `PyTorch` environment:
```bash
cd PyTorch
conda create -n GDL_PyTorch python==3.9
conda activate GDL_PyTorch
pip install -r requirements.txt
```
NOTE: If you're using PyTorch on WSL, please add `export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH` to `~/.bashrc` to avoid kernel restarting problems.<br>
<br> 
Configure `JAX` environment:
```bash
cd JAX_FLAX
conda create -n GDL_JAX python==3.9
conda activate GDL_JAX
pip install -r requirements.txt
```

`.ipynb` is the extension of the python notebook; I use [Jupyter Lab](https://jupyter.org/install) to run the notebooks in this repository.

## **Model list**
### *Chapter 2 Deep Learning*
- MLP ([Pytorch](https://github.com/terrence-ou/Generative-Deep-Learning-2nd-Edition-PyTorch-JAX/blob/main/PyTorch/chapter_02_deeplearning/01_MLP.ipynb) | [JAX](https://github.com/terrence-ou/Generative-Deep-Learning-2nd-Edition-PyTorch-JAX/blob/main/JAX_FLAX/chapter_02_deeplearning/01_MLP.ipynb)) 
- CNN ([Pytorch](https://github.com/terrence-ou/Generative-Deep-Learning-2nd-Edition-PyTorch-JAX/blob/main/PyTorch/chapter_02_deeplearning/02_CNN.ipynb) | [JAX](https://github.com/terrence-ou/Generative-Deep-Learning-2nd-Edition-PyTorch-JAX/blob/main/JAX_FLAX/chapter_02_deeplearning/02_CNN.ipynb))
### *Chapter 3 Variational AutoEncoder*
- AutoEncoder ([Pytorch](https://github.com/terrence-ou/Generative-Deep-Learning-2nd-Edition-PyTorch-JAX/blob/main/PyTorch/chapter_03_vae/01_autoencoder.ipynb) | [JAX](https://github.com/terrence-ou/Generative-Deep-Learning-2nd-Edition-PyTorch-JAX/blob/main/JAX_FLAX/chapter_03_vae/01_autoencoder.ipynb))
- VAE (FashionMNIST) ([Pytorch](https://github.com/terrence-ou/Generative-Deep-Learning-2nd-Edition-PyTorch-JAX/blob/main/PyTorch/chapter_03_vae/02_vae_fashion.ipynb) | [JAX](https://github.com/terrence-ou/Generative-Deep-Learning-2nd-Edition-PyTorch-JAX/blob/main/JAX_FLAX/chapter_03_vae/02_vae_fashion.ipynb))
- VAE (CelebA Dataset) ([Pytorch](https://github.com/terrence-ou/Generative-Deep-Learning-2nd-Edition-PyTorch-JAX/blob/main/PyTorch/chapter_03_vae/03_vae_face.ipynb) | [JAX](https://github.com/terrence-ou/Generative-Deep-Learning-2nd-Edition-PyTorch-JAX/blob/main/JAX_FLAX/chapter_03_vae/03_vae_faces.ipynb))
### *Chapter 4 Generative Adversarial Networks*
- DCGAN ([Pytorch](https://github.com/terrence-ou/Generative-Deep-Learning-2nd-Edition-PyTorch-JAX/blob/main/PyTorch/chapter_04_gan/01_dcgan.ipynb) | [JAX](https://github.com/terrence-ou/Generative-Deep-Learning-2nd-Edition-PyTorch-JAX/blob/main/JAX_FLAX/chapter_04_gan/01_dcgan.ipynb))
- WGAN-GP ([Pytorch](https://github.com/terrence-ou/Generative-Deep-Learning-2nd-Edition-PyTorch-JAX/blob/main/PyTorch/chapter_04_gan/02_wgan_gp.ipynb) | [JAX](https://github.com/terrence-ou/Generative-Deep-Learning-2nd-Edition-PyTorch-JAX/blob/main/JAX_FLAX/chapter_04_gan/02_wgan_gp.ipynb))
- Conditional GAN ([Pytorch](https://github.com/terrence-ou/Generative-Deep-Learning-2nd-Edition-PyTorch-JAX/blob/main/PyTorch/chapter_04_gan/03_cgan.ipynb) |[JAX](https://github.com/terrence-ou/Generative-Deep-Learning-2nd-Edition-PyTorch-JAX/blob/main/JAX_FLAX/chapter_04_gan/03_cgan.ipynb))
### *Chapter 5 Autogressive Model*
- LSTM([Pytorch](https://github.com/terrence-ou/Generative-Deep-Learning-2nd-Edition-PyTorch-JAX/blob/main/PyTorch/chapter_05_autogressive/01_lstm.ipynb) | JAX)
