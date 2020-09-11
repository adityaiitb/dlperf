# ResNet50

The ResNet50 model was obtained from [Torchvision][1]. It was then
annotated with NVTX markers and modified to enable profiling using
Nvidia's PyTorch Profiler [PyProf][2]. We use a synthetic dataset.

## Install Nvidia's PyTorch Profiler [PyProf][2]

```sh
git clone https://github.com/NVIDIA/PyProf.git
cd PyProf
pip3 install .
```

## Profile using NVProf or NSys to obtain a SQLite3 database
```sh
nvprof -fo resnet.sqlite --profile-from-start off ./resnet.py
```

## Parse the database and then obtain the profile
```sh
python3 -m pyprof.parse resnet.sqlite > resnet.dict
python3 -m pyprof.prof resnet.dict
```

[1]: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
[2]: https://github.com/NVIDIA/PyProf
