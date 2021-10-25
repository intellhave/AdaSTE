# Train binarized ResNets on CIFAR-10 with BinaryConnect
LR=0.01
DEPTH=50
i=0
dataset='cifar100'
python3 main_binary_reg.py --model resnet --model_config "{'depth': $DEPTH}" --save resnet"$DEPTH"_"$dataset"_ST --dataset $dataset --gpu 0 --batch-size 128 --epochs 300 --binary_reg 1.0 --tb_dir tb/CIFAR_resnet"$DEPTH"_ST --optimizer Adam --lr $LR --binary_regime --projection_mode lazy --freeze_epoch 200 --print-freq 100 
#--resume /home/intellhave/Work/NetQuantz/results/resnet20_ST/checkpoint.pth.tar
