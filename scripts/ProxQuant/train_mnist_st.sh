# Train binarized ResNets on CIFAR-10 with BinaryConnect
python3 main_binary_reg.py --model simple_cnn --model_config "{'depth': 0}" --save simple_cnn --dataset mnist --gpu 0 --batch-size 128 --epochs 300 --binary_reg 1.0 --tb_dir tb/mnist_ST --optimizer SGD --lr 0.01 --binary_regime --projection_mode lazy --freeze_epoch 2 &
