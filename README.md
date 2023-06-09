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
Configure `JAX` environment:
```bash
cd JAX_FLAX
conda create -n GDL_JAX python==3.9
conda activate GDL_JAX
pip install -r requirements.txt
```

`.ipynb` is the extension of the python notebook; I use [Jupyter Lab](https://jupyter.org/install) to run the notebooks in this repository.
