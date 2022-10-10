# Environment variables
```bash
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
NGPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F"," '{print NF}')
PORT=${PORT:-9001}
```

# Prepare your dataset
CIFAR dataset will be downloaded automatically, or you can specify the location of your dataset root with --data.

ImageNet dataset should be in the form of lmdb. See [https://github.com/xunge/pytorch_lmdb_imagenet](!https://github.com/xunge/pytorch_lmdb_imagenet).

Folder structure:

```
root
`-- train.py
`-- data
    `-- cifar-100-python
    `-- cifar-10-batches-py
    `-- I100
        `-- train.lmdb
        `-- val.lmdb
    `-- I1K
        `-- train.lmdb
        `-- val.lmdb
`-- [other files and directories]
```

# Cifar-10

1. Training

    ```bash
    python -m torch.distributed.launch --nproc_per_node $NGPUS --master_port $PORT train.py --data ./data --cifar --num_classes 10 --aa --from_ $MODEL_FILE --import_ $MODEL_FUNCTION_NAME
    ```

# Cifar-100

1. Training

    ```bash
    python -m torch.distributed.launch --nproc_per_node $NGPUS --master_port $PORT train.py --data ./data --cifar --num_classes 100 --aa --from_ $MODEL_FILE --import_ $MODEL_FUNCTION_NAME
    ```

# ImageNet-100

1. Training

    ```bash
    python -m torch.distributed.launch --nproc_per_node $NGPUS --master_port $PORT train.py --data ./data/I100 --num_classes 100 --from_ $MODEL_FILE --import_ $MODEL_FUNCTION_NAME
    ```

2. Evaluate
   
   ```bash
   # this is an example
   python evaluate.py --data ./data/I100 --num_classes 100 --from_ model/repnet.py --import_ repnet50 --evaluate_model logs/se_rep/model_best_rep.pth.tar
   ```

# ImageNet-1K

1. Training

    ```bash
    python -m torch.distributed.launch --nproc_per_node $NGPUS --master_port $PORT train.py --data ./data/I1K --num_classes 1000 --from_ $MODEL_FILE --import_ $MODEL_FUNCTION_NAME
    ```