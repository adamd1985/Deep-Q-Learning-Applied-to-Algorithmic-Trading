# Deep Q-Learning Applied to Algorithmic Trading

The notebook with the experiments is available [here](./drl_trading.ipynb)
The study is a replication and enhancement of the DQN by Theate, Thibaut and Ernst, Damien(2020).

# Dataset

The data is provided by me in the [data directory](./data)

# GPU and Linux

## Windows WSL

It's recommend to run this on a linux system. If you are on windows, install WSL on an admin powershell: `wsl --install`

Update your distribution: `sudo apt-get update` and `sudo apt-get install wget ca-certificates`. From your WSL cli, run `code .` and it will open a VSCode for you. Install the VSCode WSL extension for a seamless developer experience, see their tutorial [here](https://code.visualstudio.com/docs/remote/wsl-tutorial).

## Cuda and GPUs

Install the NVidia [drivers](https://www.nvidia.com/Download/index.aspx) on your Linux box.
Check if installation was good: `nvidia-smi`

Install TensorFlow's cuda framework: `pip install tensorflow[and-cuda]` and verify the installation: `python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`


## Environment

Create the environment
```bash
conda create -n drl_ta python=3.11 && conda activate drl_ta && conda install -c conda-forge tensorflow-gpu cudatoolkit=11.8 cudnn yfinance matplotlib scipy && pip install --user tf-agents[reverb] ta tqdm pyarrow
```
or import the env from the JSON:
```bash
conda env create -f ./env.yml
```

if you get a TypeAlias error causing the Kernel to fail, upgrade it:
```bash
pip install --upgrade typing_extensions
```

## Install FinRL

FinRL was one of the many inspirations. If you wish to try it, follow their installation [guide](https://finrl.readthedocs.io/en/latest/start/installation.html).

```bash
sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev libgl1-mesa-glx
pip install git+https://github.com/AI4Finance-Foundation/FinRL.git
```

Try it out:
```bash
git clone https://github.com/AI4Finance-Foundation/FinRL.git
cd FinRL
python tutorials/FinRL_StockTrading_DQN.py
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