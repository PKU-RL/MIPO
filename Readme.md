## Installation:

See [GATA](https://github.com/xingdi-eric-yuan/GATA-public) for more detials

## Execution:

```shell
CUBLAS_WORKSPACE_CONFIG=:16:8 CUDA_VISIBLE_DEVICES=0 python3 train.py configs/train_rl.yaml -p general.random_seed={SEED} rl.difficulty_level={DIFF}
```