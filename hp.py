import os
import yaml

hyperparameters_virtual={
'anchor': 50, # all images/frames in experiments are extracted from the range [anchor_idx, 300], where 300 is the maximum length of long/full videos
'batch_size': 40, # larger bs because I increased numEclass from 5 to 50
'dataset_name': 'vir_poppy',
'dilation': 8, # all images/frames in experiments are extracted from original long/full videos following the rate 1:dilation, which is 1:10
'epochs': 200, # just epochs
'f_size': 10, # the number of frames/images will Poppy take into consideration for its prediction
'latent_size': 2048,
'lr_m1': 0.0001,
'prediction_span': 1, # at timestep t, Poppy predicts its state at timestep t+prediction_span, which is t+2
}

hyperparameters_real={
'batch_size': 40, # larger bs because I increased numEclass from 5 to 50
'dataset_name': 'real_poppy',
'dilation': 1, # camputered by a real camera
'epochs': 200, # just epochs
'f_size': 10, # the number of frames/images will Poppy take into consideration for its prediction
'latent_size': 256,
'lr_m1': 0.0001,
'prediction_span': 1, # at timestep t, Poppy predicts its state at timestep t+prediction_span, which is t+2
}

# with open(os.path.join(os.getcwd(), 'checkpoints', 'debugging', 'HyperParam.yml'), 'w') as outfile:
#         yaml.dump(hyperparameters_real, outfile, default_flow_style=False)