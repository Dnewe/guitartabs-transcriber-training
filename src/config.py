# DATA
TRAIN_PROP = 0.8 # proportion of train data compared to dev data
MULTICLASS_LABELS = True
SINGLECLASS_LABELS = False

# position of multi-label classification labels in data CSV
Y_MULTI_START = 0
Y_MULTI_END = 6
# position of single-label classification labels in data CSV
Y_SINGLE_START = 6
Y_SINGLE_END = 12
# starting position of elements in data CSV
X_STARTINDEX = 12 

# LEARNING HYPERPARAMETERS
ITERATIONS = 200
ALPHA = 0.3

# NEURON NETWORK
SIZE_LAYER1 = 30