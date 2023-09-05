from training_gaga import training_m1
from testing_gaga import testing_m1
from utils.visualization_training_gaga import vis_training_m1
from utils.visualization_testing_gaga import vis_testing_m1
from utils.summary_plot import plot
from hp import hyperparameters_virtual, hyperparameters_real 
from m2_sup import m2_sup
import argparse


parser = argparse.ArgumentParser(description='Process user defined hyperparameters')
parser.add_argument('dataset', type=str, help='name of dataset')
parser.add_argument('-ps', nargs='+', type=int, help='a list of prediction spans')
parser.add_argument('-ls', nargs='+', type=int, help='a list of latent sizes')

# args = parser.parse_args(['vir', '-ps', '1', '2', '-ls', '11', '22'])

args = parser.parse_args()

# Hyperparameters
if args.dataset == 'vir':
    HpParams = hyperparameters_virtual
elif args.dataset == 'real':
    HpParams = hyperparameters_real

for ps in set(args.ps):
    for ls in set(args.ls):
        # checkpoint_name = 'gaga_' + args.dataset + f'_ps{ps}_{ls}'
        # print(checkpoint_name)
        HpParams['prediction_span'] = int(ps)
        HpParams['latent_size'] = int(ls)

        ################## model 1 ##################
        checkpoint_name = training_m1(HpParams, debugging=False)
        testing_m1(checkpoint_name)
        vis_training_m1(checkpoint_name)
        vis_testing_m1(checkpoint_name)

        ################## model 2 ##################
        m2_sup(checkpoint_name)
plot(args.dataset)