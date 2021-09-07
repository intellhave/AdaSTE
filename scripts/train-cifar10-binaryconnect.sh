# Train binarized ResNets on CIFAR-10 with BinaryConnect
LR=0.01
DEPTH=20
i=0
python3 main_binary_reg.py --model resnet --model_config "{'depth': $DEPTH}" --save resnet"$DEPTH"_bc_Adam_run_$i --dataset cifar10 --gpu 0 --batch-size 128 --epochs 300 --binary_reg 1.0 --tb_dir tb/CIFAR_resnet"$DEPTH"_ST --optimizer Adam --lr $LR --binary_regime --projection_mode lazy --freeze_epoch 200 &
