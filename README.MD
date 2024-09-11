# Deep Q-Learning Applied to Algorithmic Trading

The notebook with the experiments is available [here](./drl_trading.ipynb)
The study is a replication and enhancement of the DQN by Theate, Thibaut and Ernst, Damien(2020).

# Dataset

The Data is provided by me in the [data directory](./data)

# GPU and Linux

## Windows WSL

It's recommend to run this on a linux system.

If you are on windows, install WSL on an admin powershell: `wsl --install`

update your distribution: `sudo apt-get update` and `sudo apt-get install wget ca-certificates`. From your WSL cli, run `code .` and it will open a VSCode for you. Install the VSCode WSL extension for a seamless developer experience, see their tutorial [here](https://code.visualstudio.com/docs/remote/wsl-tutorial).

## Cuda and GPUs

Install the NVidia [drivers](https://www.nvidia.com/Download/index.aspx) on your Linux box.
Check if installation was good: `nvidia-smi`

Install TensorFlow's cuda framework: `pip install tensorflow[and-cuda]` and verify the installation: `python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`


##

Environment

Add these conda repos:
```bash
conda config --add channels pytorch && conda config --add channels conda-forge && conda config --add channels scikit-learn
```

Create an env from the JSON:
```bash
conda env create -f ./conda_env.yml
```

# Credits and Citations

The DQN model was inspired by the paper:

```BibTeX
@article{theate2021application,
  title={An application of deep reinforcement learning to algorithmic trading},
  author={Th{\'e}ate, Thibaut and Ernst, Damien},
  journal={Expert Systems with Applications},
  volume={173},
  pages={114632},
  year={2021},
  publisher={Elsevier}
}
```