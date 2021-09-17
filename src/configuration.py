import configargparse


def parser_args():
    parser = configargparse.ArgParser(description='PyTorch ImageNet Training')
    parser.add('-c', '--config', required=True,
               is_config_file=True, help='config file')
    ### dataset
    parser.add_argument('--data-type', type=str, choices=('miniImageNet', 'tieredImageNet'),
                        default='miniImageNet',
                        help='Data type')
    parser.add_argument('--data', metavar='DIR', help='path to dataset')
    parser.add_argument('--num-classes', type=int, default=64,
                        help='use all data to train the network')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--disable-train-augment', action='store_true',
                        help='disable training augmentation')
    parser.add_argument('--disable-random-resize', action='store_true',
                        help='disable random resizing')
    parser.add_argument('--enlarge', action='store_true',
                        help='enlarge the image size then center crop')
    ### network setting
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                        help='network architecture')
    parser.add_argument('--scheduler', default='step', choices=('step', 'multi_step', 'cosine'),
                        help='scheduler, the detail is shown in train.py')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--lr-stepsize', default=30, type=int,
                        help='learning rate decay step size ("step" scheduler) (default: 30)')
    parser.add_argument('--lr-gamma', default=0.1, type=float,
                        help='gamma for learning rate decay (default: 0.1)')
    parser.add_argument('--optimizer', default='SGD', choices=('SGD', 'Adam'))
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--nesterov', action='store_true',
                        help='use nesterov for SGD, disable it in default')
    ### meta val setting
    parser.add_argument('--meta-test-iter', type=int, default=10000,
                        help='number of iterations for meta test')
    parser.add_argument('--meta-val-iter', type=int, default=500,
                        help='number of iterations for meta val')
    parser.add_argument('--meta-val-way', type=int, default=5,
                        help='number of ways for meta val/test')
    parser.add_argument('--meta-val-shot', type=int, default=5,
                        help='number of shots for meta val/test')
    parser.add_argument('--meta-val-query', type=int, default=15,
                        help='number of queries for meta val/test')
    parser.add_argument('--meta-val-interval', type=int, default=10,
                        help='do mate val in every D epochs')
    parser.add_argument('--meta-val-metric', type=str, choices=('euclidean', 'cosine', 'l1', 'l2'),
                        default='euclidean',
                        help='meta-val/test evaluate metric')
    parser.add_argument('--num_NN', type=int, default=1,
                        help='number of nearest neighbors, set this number >1 when do kNN')
    parser.add_argument('--eval-fc', action='store_true',
                        help='do evaluate with final fc layer.')
    ### meta train setting
    parser.add_argument('--do-meta-train', action='store_true',
                        help='do prototypical training')
    parser.add_argument('--meta-train-iter', type=int, default=100,
                        help='number of iterations for meta val')
    parser.add_argument('--meta-train-way', type=int, default=30,
                        help='number of ways for meta val')
    parser.add_argument('--meta-train-shot', type=int, default=1,
                        help='number of shots for meta val')
    parser.add_argument('--meta-train-query', type=int, default=15,
                        help='number of queries for meta val')
    parser.add_argument('--meta-train-metric', type=str, choices=('euclidean', 'cosine', 'l1', 'l2'),
                        default='euclidean',
                        help='meta-train evaluate metric')


    # Distance compare argument
    parser.add_argument('--analyze', action='store_true',
                        help='compare distance between same class and different class')
    parser.add_argument('--anlz-interval', type=int, default=1,
                        help='do distance comparison in every D epochs')
    parser.add_argument('--anlz-iter',type = int, default = 1000,
                        help = 'number of iterations for distance compare')
    parser.add_argument('--anlz-class',type = int, default = 2,
                        help = 'number of classes to compare, need more than 1')
    parser.add_argument('--anlz-base',type = int, default = 1,
                        help = 'number of base image')
    parser.add_argument('--anlz-same',type = int, default = 1,
                        help = 'number of same class image')
    parser.add_argument('--anlz-diff',type = int, default = 1,
                        help = 'number of different class image')
    parser.add_argument('--anlz-metric', type=str, choices=('euclidean', 'cosine', 'l1', 'l2'),
                        default='euclidean',
                        help='feature distance compare metric')
    parser.add_argument('--anlz-rand', action='store_true',
                        help='sample random class for each analyzation when true')
    parser.add_argument('--anlz-FC', action='store_true',
                        help='compare features passed through FC Layer')

    # Triplet train
    parser.add_argument('--triplet-batch',type = int, default = 300,
                        help = 'steps for each iteration')
    parser.add_argument('--triplet-iter',type = int, default = 150,
                        help = 'number of iterations for triplet training')
    parser.add_argument('--triplet-rand', action='store_true',
                        help='sample random class for each epoch when true')
    parser.add_argument('--triplet-base',type = int, default = 1,
                        help = 'number of same class image')
    parser.add_argument('--triplet-same',type = int, default = 1,
                        help = 'number of same class image')
    parser.add_argument('--triplet-diff',type = int, default = 1,
                        help = 'number of different class image')
    parser.add_argument('--triplet-metric', type=str, choices=('euclidean', 'cosine', 'l1', 'l2'),
                        default='euclidean',
                        help='feature distance compare metric')

    parser.add_argument('--mix-train', action='store_true',
                        help='use both Triplet Loss and Cross Entropy for training')
    parser.add_argument('--ce-train', action='store_true',
                        help='use Cross Entropy for training')
    parser.add_argument('--trp-train', action='store_true',
                        help='use Triplet Loss for training')

    ### others
    parser.add_argument('--split-dir', default=None, type=str,
                        help='path to the folder stored split files.')
    parser.add_argument('--save-path', default='result/default', type=str,
                        help='path to folder stored the log and checkpoint')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--disable-tqdm', action='store_true',
                        help='disable tqdm.')
    parser.add_argument('--print-freq', '-p', default=150, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--evaluate', action='store_true',
                        help='evaluate the final result')
    parser.add_argument('--pretrain', type=str, default=None,
                        help='path to the pretrained model, used for fine-tuning')
    parser.add_argument('--logname', type=str, default='training.log',
                        help='name of the training log file')


    return parser.parse_args()
