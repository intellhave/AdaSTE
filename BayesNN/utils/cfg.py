import argparse  

def get_parser():
    parser = argparse.ArgumentParser(description = "FenBP")

    parser.add_argument('--model', type = str, default = 'RESNET18', help = 'Model Name: RESNET, VGG')
    parser.add_argument('--dataset', type = str, default = 'cifar10', help = 'Datasetes')
    parser.add_argument('--bnmomentum', type=float, default=0.2, help='BN layer momentum value')
    # Optimization parameters
    parser.add_argument('--optim', type=str, default='FenBP', help='Optimizer: BayesBiNN, STE, Adam')
    parser.add_argument('--val-split', type=float, default=0.1, help='Random validation set ratio')
    parser.add_argument('--criterion', type=str, default='cross-entropy', help='loss funcion: square-hinge or cross-entropy')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--train-samples', type=int,default=1, metavar='N',
                        help='number of Monte Carlo samples used in BayesBiNN (default: 1), if 0, point estimate using mean')
    parser.add_argument('--test-samples', type=int,default=0, metavar='N',
                        help='number of Monte Carlo samples used in evaluation for BayesBiNN (default: 1)')
    parser.add_argument('--epochs', type=int, default=700, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default= 3e-4, metavar='LR',
                        help='learning rate (default: 0.0003)')
    parser.add_argument('--lr-end', type=float, default= 1e-16, metavar='LR',
                        help='end learning rate (default: 0.01)')
    parser.add_argument('--lr-decay', type=float, default= 0.9, metavar='LR-decay',
                        help='learning rated decay factor for each epoch (default: 0.9)')
    parser.add_argument('--decay-steps', type=int, default=1, metavar='N',
                        help='LR rate decay steps (default: 1)')   

    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='BayesBiNN momentum (default: 0.9)')
    parser.add_argument('--data-augmentation', action='store_true', default=True, help='Enable data augmentation')
    # Logging parameters
    parser.add_argument('--log-interval', type=int, default=400, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--experiment-id', type=int, default=0, help='Experiment ID for log files (int)')
    # Computation parameters
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=100, metavar='S',
                        help='random seed (default: 10)')

    parser.add_argument('--lrschedular', type=str, default='Cosine', help='Mstep,Expo,Cosine')


    parser.add_argument('--trainset_scale', type=int, default=10, metavar='N',
                        help='scale of the training set used in data augmentation')


    parser.add_argument('--lamda', type=float, default= 10, metavar='lamda-init',
                        help='initial mean value of the natural parameter lamda(default: 10)')

    parser.add_argument('--lamda-std', type=float, default= 0, metavar='lamda-init',
                        help='linitial std value of the natural parameter lamda(default: 0)')

    parser.add_argument('--temperature', type=float, default= 1e-10, metavar='temperature',
                        help='temperature for BayesBiNN')

    parser.add_argument('--bn-affine', type=float, default= 0, metavar='bn-affine',
                        help='whether there is bn learnable parameters, 1: learnable, 0: no (default: 0)')

    parser.add_argument('--data_path', type=str, default='./data/', help='path to dataset' )
    parser.add_argument('--beta_inc_rate', type=float, default=1.05)
    parser.add_argument('--init_beta', type=float, default=0.01)
    parser.add_argument('--alpha', type=float, default=0.01)
    parser.add_argument('--weight-decay', type=float, default=1.0)
    parser.add_argument('--resume_path', type=str, default='')


    return parser

