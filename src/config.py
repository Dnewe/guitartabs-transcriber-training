import seaborn as sns


# DATA
TRAIN_PROP = 0.005 # proportion of train data compared to dev data
MAX_POSITION = 3
STRINGS = 6

# position of labels in data CSV
Y_strings = [0,1,2,3,4,5]
Y_positions = [6,7,8,9,10,11]
Y_START = 0
Y_END = 12
# starting position of elements in data CSV
X_STARTINDEX = 12

# LEARNING HYPERPARAMETERS
ITERATIONS = 20000
ALPHA = 0.01

# NEURON NETWORK
SIZE_LAYER1 = 100

# OTHERS
SHOW_ACCURACY_TIME = 10

# PLOT CONFIG
sns.set_theme(style="darkgrid")