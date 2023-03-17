import argparse
from random import choices

parser = argparse.ArgumentParser(description='Sal Based Image Enhancement Options')
parser.add_argument('--expdscp', default='DESCRIPTION')

parser.add_argument('--gpu_ids',default='0') # -1 for cpu
parser.add_argument('--seed',type=int, default=543)
parser.add_argument('--num_threads',type=int, default=4)

parser.add_argument('--is_train', type=bool, default=1)
parser.add_argument('--shuffle',type=bool, default=True)

parser.add_argument('--nops', type=int ,default=4, help='Number of operations to choose from')
parser.add_argument('--crop_size',type=int, default=384)
parser.add_argument('--load_size',type=int, default=384)
parser.add_argument('--log_interval',type=int, default=50)
parser.add_argument('--val_interval',type=int, default=500)
parser.add_argument('--savemodel_interval',type=int, default=5000)

parser.add_argument('--epochs',type=int, default=10)
parser.add_argument('--lr_parameters',type=float, default=0.000001)
parser.add_argument('--lr_d',type=float, default=0.00001)
parser.add_argument('--batch_size',type=int, default=1)

parser.add_argument('--beta_r',type=float, default=0.1)
parser.add_argument('--w_sal',type=float, default=5)
parser.add_argument('--human_weight_gan', type=float, default=10)
parser.add_argument('--sal_loss_type', choices=['percentage','percentage_increase'], default='percentage')

parser.add_argument('--init_parameternet_weights', default=None)

parser.add_argument('--rgb_root', required=True)
parser.add_argument('--mask_root', required=True)
parser.add_argument('--result_for_decrease', type=int, default=1)

parser.add_argument('--result_path', default='./results/mydataset')

args = parser.parse_args()
