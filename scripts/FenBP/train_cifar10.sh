LR=0.01
DEPTH=20
python my_main.py  --model resnet --model_config "{'depth': $DEPTH}" --save resnet"$DEPTH"_FenBP --dataset cifar10 --batch-size 128 --gpu 0 --epochs 300 --tb_dir tb/CIFAR10_FenBP --optimizer Adam --lr $LR --print-freq 390 --delta_decrease_epoch 1

