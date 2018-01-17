
# Import pandas
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np




# Read in white wine data
white = pd.read_csv("data/winequality-white.csv", sep=';')

# Read in red wine data
red = pd.read_csv("data/winequality-red.csv", sep=';')


np.random.seed(570)

redlabels = np.unique(red['quality'])
whitelabels = np.unique(white['quality'])



# Add `type` column to `red` with value 1
red['type'] = 1

# Add `type` column to `white` with value 0
white['type'] = 0

# Append `white` to `red`
wines = red.append(white, ignore_index=True)

corr = wines.corr()
sns.heatmap(corr,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
plt.show()