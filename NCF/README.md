# Neural Collaborative Filtering

The code in this directory was obtained from [Nvidia Deep Learning
Examples][1]. It was then modified and annotated with NVTX markers to
enable profiling using Nvidia's PyTorch Profiler [PyProf][2].

## Install Nvidia's PyTorch Profiler [PyProf][2]

```sh
git clone https://github.com/NVIDIA/PyProf.git
cd PyProf
pip3 install .
```

## Download dataset

This should take less than a minute.

```sh
./prepare_dataset.sh
```

## Profile using NVProf or NSys to obtain a SQLite3 database

```sh
nvprof -fo ncf%p.sqlite \
	--profile-child-processes \
	--profile-from-start off \
	python -m torch.distributed.launch \
	--nproc_per_node=1 \
	--use_env ncf.py \
	--data ./data/cache/ml-20m \
	-e 1
```

```sh
nsys profile \
	-f true \
	-o ncf \
	--export sqlite \
	-c cudaProfilerApi \
	-s none \
	--stop-on-range-end true \
	python -m torch.distributed.launch \
	--nproc_per_node=1 \
	--use_env ncf.py \
	--data ./data/cache/ml-20m \
	-e 1
```

## Parse the database and then obtain the profile

```sh
python3 -m pyprof.parse ncf.sqlite > ncf.dict
python3 -m pyprof.prof ncf.dict
```

[1]: https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Recommendation/NCF
[2]: https://github.com/NVIDIA/PyProf

