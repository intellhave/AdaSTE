# model=simple_mlp
DEPTH=20
python my_main.py  --model resnet --model_config "{'depth': $DEPTH}" --save resnet"$DEPTH" --dataset cifar10 --batch-size 128 --gpu 0 --epochs 2 --tb_dir tb/CIFAR10_FenBP

