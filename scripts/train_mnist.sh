python my_main.py --model simple_cnn --model_config "{'depth': 20}" --save resnet"$DEPTH" --dataset mnist --batch-size 128 --gpu 0 --epochs 200 --tb_dir tb/MNIST_CNN_FenBP --optimizer SGD
# python my_main.py --model resnet --model_config "{'depth': 20}" --save resnet"$DEPTH" --dataset cifar10 --batch-size 128 --gpu 0 --epochs 200 --tb_dir tb/CNN_FenBP --optimizer SGD
