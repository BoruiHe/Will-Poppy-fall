import training_regre
import training_DeepVO
from testing import testing
from utils.visualization_training import vis_tr
from utils.visualization_testing import vis_testing
from example_testing import exa_testing

mode = 'regression'
checkpoint_name = training_DeepVO.training()
testing(checkpoint_name, mode)
vis_tr(checkpoint_name)
vis_testing(checkpoint_name, mode)
exa_testing(checkpoint_name, mode)