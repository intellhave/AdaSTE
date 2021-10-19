# Train MNIST
model=simple_mlp
python main_binary_reg.py --model $model --model_config "{'depth': 20}" --save MNIST_"$model"_PQ --dataset mnist --gpu 0 --batch-size 128 --epochs 300 --reg_rate 1e-4 --tb_dir tb/MNIST_"$model"_PQ --projection_mode prox --freeze_epoch 200 &
