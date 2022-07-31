# More code and instructions will be avaliable soon

This is the official code for our paper "Unsupervised Kinematic Motion Detection for Part-segmented 3D Shape Collections", https://arxiv.org/abs/2206.08497 

# Process partnet mobility dataset:

create a folder named "data" in parallel to the "src" folder and put the PartNetMobility dataset files in the folder. 
The folder structure will be: 
data/partnet/geo
data/partnet/precomputed

python main.py --option process --category 'Category'

# Compute motion parameters:

python main.py --option train --category 'Category'

# Resolve motion annotations:

python main.py --option resolve --category 'Category'

# Compute motion prediction accuracy:

python main.py --option accuracy --category 'Category'



Other comments:

The config.py file contains a lot of adjustable hyper-parameters and configurable settings of the system, the performance of the system
can be sensitive to these hyper-parameters and settings.
