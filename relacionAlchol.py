
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


fig, ax = plt.subplots(1, 2)

ax[0].hist(red.quality, 10, facecolor='red', alpha=0.5, label="Red wine")
ax[1].hist(white.quality, 10, facecolor='white', ec="black", lw=0.5, alpha=0.5, label="White wine")

fig.subplots_adjust(left=0, right=1, bottom=0, top=0.5, hspace=0.05, wspace=1)
ax[0].set_ylim([0, 1000])
ax[0].set_xlabel("Alcohol in % Vol")
ax[0].set_ylabel("Frequency")
ax[1].set_xlabel("Alcohol in % Vol")
ax[1].set_ylabel("Frequency")
#ax[0].legend(loc='best')
#ax[1].legend(loc='best')
fig.suptitle("Distribution of Alcohol in % Vol")

plt.savefig('reporte.png')