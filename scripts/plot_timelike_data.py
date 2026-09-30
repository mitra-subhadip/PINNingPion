import pandas as pd
import matplotlib.pyplot as plt

# Initialize the plot
plt.figure(figsize=(10, 6))

# Get unique sources to plot them separately
df = pd.read_csv('../dataset/timelike_dataset.csv')
sources = df['Source'].unique()

# Define a list of markers to distinguish sources clearly
markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'X']

# Loop through each source and plot its data with error bars
for i, source in enumerate(sources):
    subset = df[df['Source'] == source]
    
    plt.errorbar(subset['q2'], subset['Fpi_sq'], yerr=subset['error'], 
                 fmt=markers[i % len(markers)], label=source, 
                 capsize=3, alpha=0.8, linestyle='none', markersize=6)

plt.xlabel(r'$Q^2$ (GeV$^2$)', fontsize=14)
plt.ylabel(r'$|F_\pi|^2$', fontsize=14)
plt.title('Timelike Pion Form Factor Data', fontsize=16)

plt.legend(title='Source', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
# plt.xlim(0.2, 1)
# plt.yscale('log')

# Save the plot
plt.savefig('timelike_dataset_plot.png', dpi=300)
plt.show()
