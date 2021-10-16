import argparse

RANDOM_SEED = 123456

INPUT_CHANNELS = 1
IM_SIZE = 28
INPUT_DIM = 3
HIDDEN_DIM = 2
OUTPUT_DIM = 2
DATASET_SIZE = 10
NUM_ITERS = 1000
NUM_EPOCHS = 1
BATCH_SIZE = 64
LEARNING_RATE = 0.0001
SAVE_DIR = './out'
GPU_ID = '0'
OPTIMIZER = 'ADAM'
MOMENTUM = 0.95
WEIGHT_DECAY = 0.0005
QUANT_LEVELS = 2    # 2: {-1, 1}, 3: {-1, 0, 1}
PROJECTION = 'SOFTMAX'  # EUCLIDEAN
DATASET = 'MNIST'       # CIFAR10, CIFAR100, TINYIMAGENET200, IMAGENET1000
ARCHITECTURE = 'MLP'    # LENET300, LENET5, VGG16, RESNET18
ROUNDING = 'ARGMAX'     # ICM
EVAL_SET = 'TEST'   # TEST
VAL_SET = 'TRAIN'   # VAL
LOSS_FUNCTION = 'HINGE'   # CROSSENTROPY
METHOD = 'CONTINUOUS'
BETA_SCALE = 1.2
LR_SCALE = 1.
LR_INTERVAL = 100
BETA_INTERVAL = 500
LR_DECAY = 'STEP'   # EXP, MSTEP
LOG_INTERVAL = 300
PR_INTERVAL = 100
SAVE_NAME = ''          # best model file
DATA_PATH = '../Datasets'
PRETRAINED_MODEL = ''   # pre-trained model file
EVAL = ''       # trained model to evaluate
RESUME = ''         # checkpoint file to resume


def get_arguments():
    """Parse all the arguments provided from the CLI.
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Script for MD based NN Quantization.")

    ### Experiment Settings
    parser.add_argument("--method", type=str, default=METHOD,
                        help="Method to run [CONTINUOUS, BC, SOFTMAX_PROJECTION, TANH_PROJECTION, MD_SOFTMAX, MD_TANH].")
    parser.add_argument("--loss-function", type=str, default=LOSS_FUNCTION,
                        help="Loss function [HINGE, CROSSENTROPY].")
    parser.add_argument("--zeroone", action="store_true", help="Flag to set Q_l = {0,1} in case of binary")
    parser.add_argument("--tanh", action="store_true", help="Flag to perform tanh projection on model parameters for binary quantization")
    parser.add_argument("--full-ste", action="store_true", help="Flag for full-STE")
    parser.add_argument("--quant-levels", type=int, default=QUANT_LEVELS,
                        help="Quantization levels: {2: {-1, 1}, 3: {-1, 0, 1}}.")
    parser.add_argument("--projection", type=str, default=PROJECTION,
                        help="Type of projection [SOFTMAX, EUCLIDEAN, ARGMAX].")
    parser.add_argument("--dataset", type=str, default=DATASET,
                        help="Type of architecture [MNIST, CIFAR10, CIFAR100, TINYIMAGENET200, IMAGENET1000].")
    parser.add_argument("--architecture", type=str, default=ARCHITECTURE,
                        help="Type of architecture [VGG16, RESNET18].")
    parser.add_argument("--rounding", type=str, default=ROUNDING, help="Type of rounding [ARGMAX, ICM].")

    ### Optimizer Settings
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE, help="Learning rate.")
    parser.add_argument("--lr-scale", type=float, default=LR_SCALE, help="Scale to multiply learning rate.")
    parser.add_argument("--lr-interval", type=str, default=LR_INTERVAL,
                        help="No of iterations before changing lr.")
    parser.add_argument("--lr-decay", type=str, default=LR_DECAY,
                        help="LR decay type [STEP, EXP, MSTEP].")
    parser.add_argument("--nesterov", action="store_true", help="Flag to use Nesterov momentum.")
    parser.add_argument("--optimizer", type=str, default=OPTIMIZER,
                        help="Type of optimizer [SGD, ADAM, MD_SOFTMAX_ADAM, MD_TANH_ADAM].")
    parser.add_argument("--momentum", type=float, default=MOMENTUM, help="Momentum.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY, help="Weight decay.")
    parser.add_argument("--beta-scale", type=float, default=BETA_SCALE, help="Scale to multiply beta.")
    parser.add_argument("--beta-interval", type=int, default=BETA_INTERVAL,
                        help="No of iterations before changing beta.")

    ### Other Miscellaneous Settings
    parser.add_argument("--input-channels", type=int, default=INPUT_CHANNELS, help="Input channels.")
    parser.add_argument("--im-size", type=int, default=IM_SIZE, help="Image size.")
    parser.add_argument("--input-dim", type=int, default=INPUT_DIM, help="Input dimension.")
    parser.add_argument("--hidden-dim", type=int, default=HIDDEN_DIM, help="Hidden layer size.")
    parser.add_argument("--output-dim", type=int, default=OUTPUT_DIM, help="Output dimension.")
    parser.add_argument("--dataset-size", type=int, default=DATASET_SIZE, help="No of datapoints.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size.")
    parser.add_argument("--num-iters", type=int, default=NUM_ITERS, help="Number of iterations.")
    parser.add_argument("--num-epochs", type=int, default=NUM_EPOCHS, help="Number of epochs.")
    parser.add_argument("--save-dir", type=str, default=SAVE_DIR, help="Output directory.")
    parser.add_argument("--gpu-id", type=str, default=GPU_ID, help="Cuda visible device.")
    parser.add_argument("--log-interval", type=int, default=LOG_INTERVAL,
                        help="No of iterations before printing loss.")
    parser.add_argument("--pr-interval", type=int, default=PR_INTERVAL,
                        help="No of iterations before projection to the simplex.")
    parser.add_argument("--save-name", type=str, default=SAVE_NAME, help="Name to save the best model.")
    parser.add_argument("--pretrained-model", type=str, default=PRETRAINED_MODEL, help="Pretrained model to load weights.")
    parser.add_argument("--data-path", type=str, default=DATA_PATH, help="Path to store datasets.")
    parser.add_argument("--eval", type=str, default=EVAL, help="Model file to evaluate.")
    parser.add_argument("--resume", action="store_true", help="to resume training.")
    parser.add_argument("--resume_file", type=str, default=RESUME, help="Checkpoint file to resume training.")
    parser.add_argument("--use_tensorboard", action="store_true", help="Flag to Plot histogram of weights over a period of epochs using tensorboard")
    parser.add_argument("--eval-set", type=str, default=EVAL_SET, help="Dataset to evaluate [TEST, TRAIN].")
    parser.add_argument("--val-set", type=str, default=VAL_SET, help="Dataset to validate [TEST, TRAIN].")

    return parser.parse_args()


args = get_arguments()
