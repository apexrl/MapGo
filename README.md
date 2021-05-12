# MapGo: Model-Assisted Policy Optimization for Goal-Oriented Tasks

This is the TensorFlow implementation of the paper [MapGo: Model-Assisted Policy Optimization for Goal-Oriented Tasks](arxiv:xxx) accepted by IJCAI2021.



## Requirements:

```bash
pip install -r requirements.txt
```



## To run:

example:

```bash
python mapgo.py --tag=test --env=AntLocomotion-v0 --buffer_size=100000 --pi_lr=1e-3 --q_lr=1e-3 --timesteps=100 --foresight_length=30 --model_based_training=True --fgi=True

```

Hyperparameters are in ``config.py``.



## Acknowledgements

Our implementation is based on [HGG](https://github.com/Stilwell-Git/Hindsight-Goal-Generation) and [mbpo](https://github.com/JannerM/mbpo) codebase.

