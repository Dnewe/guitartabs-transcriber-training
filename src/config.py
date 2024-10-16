import seaborn as sns


# DATA
TRAIN_PROP = 0.02 # proportion of train data compared to dev data
MAX_POSITION = 14

# position of multi-label classification labels in data CSV
Y_START = 0
Y_END = 12
# starting position of elements in data CSV
X_STARTINDEX = 12

# LEARNING HYPERPARAMETERS
ITERATIONS = 10000
ALPHA = 0.01

# NEURON NETWORK
SIZE_LAYER1 = 100

# OTHERS
SHOW_ACCURACY_TIME = 20

# PLOT CONFIG
sns.set_theme(style="darkgrid")