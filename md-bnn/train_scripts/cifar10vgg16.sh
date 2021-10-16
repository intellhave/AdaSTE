#!/usr/bin/env bash

dt='CIFAR10'
sd=./out
ep=0 #0:use iters
bs=128
ar='VGG16'
lf='CROSSENTROPY'
mm=0.95
lri=30000
iters=100000
wd=0.0
gpuid=0

#### MD-tanh-S
op='ADAM'
pr='TANH'
mt='TANH_PROJECTION'
lr=0.001
lrs=0.2
bts=1.2
bti=2000

python quantized_nets.py --gpu-id $gpuid --save-dir $sd --learning-rate $lr --architecture $ar --loss-function $lf --method $mt --optimizer $op --projection $pr --dataset $dt --batch-size $bs --num-epochs $ep --momentum $mm --lr-scale $lrs --lr-interval $lri --num-iters $iters --weight-decay $wd --beta-scale $bts --beta-interval $bti --tanh --full-ste

#### MD-Softmax-S
op='ADAM'
pr='SOFTMAX'
mt='SOFTMAX_PROJECTION'
lr=0.001
lrs=0.3
bts=1.2
bti=2000

python quantized_nets.py --gpu-id $gpuid --save-dir $sd --learning-rate $lr --architecture $ar --loss-function $lf --method $mt --optimizer $op --projection $pr --dataset $dt --batch-size $bs --num-epochs $ep --momentum $mm --lr-scale $lrs --lr-interval $lri --num-iters $iters --weight-decay $wd --beta-scale $bts --beta-interval $bti --full-ste

#### MD-tanh
op='MDA_TANH_ADAM'
pr='TANH'
mt='MDA_TANH'
lr=0.001
lrs=0.3
bts=1.1
bti=1000

python quantized_nets.py --gpu-id $gpuid --save-dir $sd --learning-rate $lr --architecture $ar --loss-function $lf --method $mt --optimizer $op --projection $pr --dataset $dt --batch-size $bs --num-epochs $ep --momentum $mm --lr-scale $lrs --lr-interval $lri --num-iters $iters --weight-decay $wd --beta-scale $bts --beta-interval $bti --tanh


#### MD-Softmax
op='MDA_SOFTMAX_ADAM'
pr='SOFTMAX'
mt='MDA_SOFTMAX'
lr=0.01
lrs=0.2
bts=1.05
bti=500

python quantized_nets.py --gpu-id $gpuid --save-dir $sd --learning-rate $lr --architecture $ar --loss-function $lf --method $mt --optimizer $op --projection $pr --dataset $dt --batch-size $bs --num-epochs $ep --momentum $mm --lr-scale $lrs --lr-interval $lri --num-iters $iters --weight-decay $wd --beta-scale $bts --beta-interval $bti
