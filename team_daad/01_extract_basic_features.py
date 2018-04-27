from glob import glob
import pandas as pd
import numpy as np
import nibabel as nb

# Load the covariate file
df = pd.read_csv('pac/PAC2018_Covariates.csv')

# Get NIfTIs
niftis = sorted(glob('pac/nifti/*nii.gz'))

# Get atlas mask information
mask = nb.load('01_HarvardOxford_mask_50p_TPM.nii.gz').get_data()
with open('01_HarvardOxford_labels.txt', 'r') as f:
    labels = f.read().split('\n')[:-1]

# Create array to store new 'global' features
features_global = []
features_atlas = []

for n in niftis:

    # Load data from individual sMRI file
    img_data = nb.load(n).get_data()
    tmp = img_data[img_data != 0]

    # Add relevant information of the total GM volume to the feature array
    features_global.append([
        len(tmp),
        round(np.mean(tmp), 8),
        round(np.median(tmp), 8),
        round(np.std(tmp), 8),
        round(np.max(tmp), 8)])

    # Get mean of atlas ROI
    tmp = []
    for i in range(mask.shape[-1]):
        roi = img_data * mask[..., i]
        tmp.append(round(np.mean(roi[roi != 0]), 8))

    features_atlas.append(tmp)

    # Give feedback that subject was processed
    print(n)

# Add features_global to dataframe
features_global = np.array(features_global)

df['Tvoxels'] = features_global[:, 0]
df['Tmean'] = features_global[:, 1]
df['Tmedian'] = features_global[:, 2]
df['Tstd'] = features_global[:, 3]
df['Tmax'] = features_global[:, 4]

# Add features_atlas to dataframe
features_atlas = np.array(features_atlas)

for i, l in enumerate(labels):
    df[l] = features_atlas[:, i]

df.to_csv('pac/PAC2018_Covariates_detailed.csv', index=False)
