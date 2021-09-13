# Train a Full-Precision ResNet
DEPTH=20
python my_main.py --model resnet --model_config "{'depth': $DEPTH}" --save resnet_FenBP"$DEPTH" --dataset cifar10 --batch-size 128 --gpu 0 --epochs 300 --tb_dir tb/CIFAR_ResNet"$DEPTH"_FenBP 


# DEPTH=20
# python my_main.py --model alexnet --model_config "{'depth': $DEPTH}" --save resnet"$DEPTH" --dataset cifar10 --batch-size 128 --gpu 0 --epochs 200 --tb_dir tb/ResNet"$DEPTH"_FP
