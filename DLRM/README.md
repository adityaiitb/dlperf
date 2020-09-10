# DLRM

The code in this folder was obtained from [Nvidia Deep Learning
Examples][1]. It was then modified and annotated with NVTX markers to
enable profiling using Nvidia's PyTorch Profiler [PyProf][2]. We use a 
synthetic dataset.

## Install Nvidia's PyTorch Profiler [PyProf][2]

```sh
git clone https://github.com/NVIDIA/PyProf.git
cd PyProf
pip3 install .
```

## Profile using NVProf or NSys to obtain a SQLite3 database

```sh
nvprof -fo dlrm.sqlite \
	--profile-from-start off \
	-- python -m dlrm.scripts.main \
	--max_steps 32 \
	--synthetic_dataset
```

```sh
nsys profile \
	-f true \
	-o dlrm \
	--export sqlite \
	-c cudaProfilerApi \
	-s none \
	--stop-on-range-end true \
	python -m dlrm.scripts.main \
	--max_steps 32 \
	--synthetic_dataset
```

## Parse the database and then obtain the profile

```sh
python3 -m pyprof.parse dlrm.sqlite > dlrm.dict
python3 -m pyprof.prof dlrm.dict
```

[1]: https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Recommendation/DLRM
[2]: https://github.com/NVIDIA/PyProf
