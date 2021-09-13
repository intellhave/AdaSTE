# model=simple_mlp
model=simple_cnn
python my_main.py --model ${model} --model_config "{'depth': 20}" --save resnet"$DEPTH" --dataset mnist --batch-size 128 --gpu 0 --epochs 2 --tb_dir tb/MNIST_"$model"_FenBP
# python my_main.py --model resnet --model_config "{'depth': 20}" --save resnet"$DEPTH" --dataset cifar10 --batch-size 128 --gpu 0 --epochs 200 --tb_dir tb/CNN_FenBP --optimizer SGD
