import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('data.csv')

fig, axs = plt.subplots(4, 1, figsize=(20, 80))
for i, (n, g) in enumerate(df.groupby('Strategy')):
    axs[i].barh(g['Image & Kernel Size'], g['Value'], align='center')
    axs[i].set_title(n)
plt.tight_layout(pad=20)
plt.show()
