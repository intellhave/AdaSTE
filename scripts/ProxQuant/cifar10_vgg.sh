# Train binarized ResNets on CIFAR-10
LR=0.01
DEPTH=18
dataset=cifar10
model=vgg16

for i in 100
do
    python main_binary_reg.py --model vgg16 --model_config "{'depth': $DEPTH}" --save "$dataset"_"$model"_PQ_"$i" --dataset $dataset --gpu 0 --batch-size 128 --epochs 300 --reg_rate 1e-2 --tb_dir tb/resnet"$DEPTH"_prox_Adam_Freeze_200_run_$i --optimizer Adam --lr $LR --projection_mode prox --freeze_epoch 200  --print-freq 390 --seed $i
done

