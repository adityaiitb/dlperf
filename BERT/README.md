# BERT Pretraining

The code in this folder was obtained from [Nvidia Deep Learning
Examples][1]. It was then annotated with NVTX markers and modified to
enable profiling using Nvidia's PyTorch Profiler [PyProf][2]. We use a
synthetic dataset.

## Install Nvidia's PyTorch Profiler [PyProf][2]

```sh
git clone https://github.com/NVIDIA/PyProf.git
cd PyProf
pip3 install .
```

## Create synthetic data

This takes a nanosecond.

```sh
cd data
./create_synthetic_data.sh
```

## Modify `bert_config.json` (optional)
For the purpose of profiling and understanding the network, we reduced
the number of encoders (`num_hidden_layers`) from 24 to 2.

## Modify `./scripts/run_pretraining.sh` (optional)
For the purpose of understanding BERT GPU kernels and performance,
pretrain phase 1 is very similar to phase 2. Therefore, we modified the
script to exit after phase 1. We also modified the following options to
enable profiling on a single GPU.

```sh
train_batch_size=${1:-4}
num_gpus=${4:-1}
train_steps=${6:-10}
save_checkpoint_steps=${7:-5}
gradient_accumulation_steps=${11:-1}
```

## Profile using NVProf or NSys to obtain a SQLite3 database
```sh
nvprof -fo bert%p.sqlite --profile-child-processes bash ./scripts/run_pretraining.sh
```

## Parse the database and then obtain the profile
```sh
python3 -m pyprof.parse bert.sqlite > bert.dict
python3 -m pyprof.prof bert.dict
```

[1]: https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT
[2]: https://github.com/NVIDIA/PyProf
