## Adversarial Example Generation (for MNIST)

The code in this folder was obtained from the PyTorch tutorial on
[Adversarial Example Generation][1]. The code generates an adversarial
example for MNIST using the [Fast Gradient Sign Attack][2]. The code
was annotated with NVTX markers and modified to enable profiling using
Nvidia's PyTorch Profiler [PyProf][3]. The code downloads the MNIST
dataset (~ 117 MB) and takes only a few seconds.

## Install Nvidia's PyTorch Profiler [PyProf][2]

```sh
git clone https://github.com/NVIDIA/PyProf.git
cd PyProf
pip3 install .
```

## Profile using NVProf or NSys to obtain a SQLite3 database
```sh
nvprof -fo fgsa.sqlite fast_gradient_sign_attack.py
```

## Parse the database and then obtain the profile
```sh
python3 -m pyprof.parse fgsa.sqlite > fgsa.dict
python3 -m pyprof.prof fgsa.dict
```

[1]: https://github.com/pytorch/tutorials/blob/master/beginner_source/fgsm_tutorial.py
[2]: https://arxiv.org/abs/1412.6572
[3]: https://github.com/NVIDIA/PyProf
