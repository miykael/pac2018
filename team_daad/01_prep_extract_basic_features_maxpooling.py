from glob import glob
import pandas as pd
import numpy as np
import nibabel as nb
from skimage.measure import block_reduce

# Load the covariate file
df = pd.read_csv('data/PAC2018_Covariates.csv')

# Get NIfTIs
niftis = sorted(glob('data/nifti/*nii.gz'))

# Create array to store new 'global' features
features_global = []

# Factor to reduce by
reduc = 10
func = np.max

for n in niftis:

    # Load data from individual sMRI file
    img_data = nb.load(n).get_data()
    img_data = block_reduce(img_data, (reduc, reduc, reduc), func)

    # Add relevant information of the total GM volume to the feature array
    features_global.append(img_data)

    # Give feedback that subject was processed
    print(n, img_data.shape)

# Add features_global to dataframe
features_global = np.array(features_global)

# Only keep none zero entries
selecter = features_global.mean(0)!=0
features_global = features_global[:, selecter]

# Add features_atlas to dataframe
features_atlas = np.array(['v%06d' % r for r in range(features_global.shape[1])])

for i, l in enumerate(features_atlas):
    df[l] = features_global[:, i]

df.to_csv('data/PAC2018_Covariates_pooling_red%d_max.csv' % reduc, index=False)




# Load the covariate file
df = pd.read_csv('data/PAC2018_Covariates_Test.csv')

# Get NIfTIs
niftis = sorted(glob('data/nifti_test/*nii.gz'))

# Create array to store new 'global' features
features_global = []

# Factor to reduce by
for n in niftis:

    # Load data from individual sMRI file
    img_data = nb.load(n).get_data()
    img_data = block_reduce(img_data, (reduc, reduc, reduc), func)

    # Add relevant information of the total GM volume to the feature array
    features_global.append(img_data)

    # Give feedback that subject was processed
    print(n, img_data.shape)

# Add features_global to dataframe
features_global = np.array(features_global)

# Only keep the same none zero entries from the training set
features_global = features_global[:, selecter]

# Add features_atlas to dataframe
features_atlas = np.array(['v%06d' % r for r in range(features_global.shape[1])])

for i, l in enumerate(features_atlas):
    df[l] = features_global[:, i]

df.to_csv('data/PAC2018_Covariates_Test_pooling_red%d_max.csv' % reduc, index=False)
