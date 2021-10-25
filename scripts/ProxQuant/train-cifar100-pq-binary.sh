# Train binarized ResNets on CIFAR-10
LR=0.01
DEPTH=50
i=0
dataset=cifar100

python main_binary_reg.py --model resnet --model_config "{'depth': $DEPTH}" --save resnet"$DEPTH"_"$dataset"_PQ --dataset $dataset --gpu $i --batch-size 128 --epochs 300 --reg_rate 1e-4 --tb_dir tb/resnet"$DEPTH"_prox_Adam_Freeze_200_run_$i --optimizer Adam --lr $LR --projection_mode prox --freeze_epoch 200  --print-freq 390 

