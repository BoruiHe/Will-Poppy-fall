# Hyperparameters
hyperparameters_classification={
'k': 4, # the number of folds in cross-validation
'f_size': 10, # the number of frames/images will Poppy take into consideration for its prediction
'numEclass': 200, # the number of long/full vedioes that come from each case(falling down or vibration) used for training and validation
'epochs': 100, # just epochs
'batch_size': 40, # larger bs because I increased numEclass from 5 to 50
'lr': 0.00005, # lower lr
'dilation': 10, # all images/frames in experiments are extracted from original long/full videos following the rate 1:dilation, which is 1:10
'prediction_span': 3, # at timestep t, Poppy predicts its state at timestep t+prediction_span, which is t+2
'anchor_idx': 50, # all images/frames in experiments are extracted from the range [anchor_idx, 300], where 300 is the maximum length of long/full videos
'random seeds': [51, 654, 1000],
'TnumEclass': 60 # the number of long/full vedioes that come from each case(falling down or vibration) used for testing
}

hyperparameters_regression={
'k': 4, # the number of folds in cross-validation
'f_size': 10, # the number of frames/images will Poppy take into consideration for its prediction
'numEclass': 10, # the number of long/full vedioes that come from each case(falling down or vibration) used for training and validation
'epochs': 200, # just epochs
'batch_size': 40, # larger bs because I increased numEclass from 5 to 50
'lr': 0.00005, # lower lr
'dilation': 10, # all images/frames in experiments are extracted from original long/full videos following the rate 1:dilation, which is 1:10
'prediction_span': 2, # at timestep t, Poppy predicts its state at timestep t+prediction_span, which is t+2
'anchor_idx': 50, # all images/frames in experiments are extracted from the range [anchor_idx, 300], where 300 is the maximum length of long/full videos
'random seeds': [51, 654, 1000],
'TnumEclass': 60 # the number of long/full vedioes that come from each case(falling down or vibration) used for testing
}