# Train binarized ResNets on CIFAR-10
LR=0.01
DEPTH=18

for i in 0
do
    python main_binary_reg.py --model resnet --resume results/resnet"$DEPTH"_PQ --model_config "{'depth': $DEPTH}" --save resnet"$DEPTH"_PQ --dataset cifar10 --gpu $i --batch-size 128 --epochs 300 --reg_rate 1e-2 --tb_dir tb/resnet"$DEPTH"_prox_Adam_Freeze_200_run_$i --optimizer Adam --lr $LR --projection_mode prox --freeze_epoch 200  --print-freq 390 
done

