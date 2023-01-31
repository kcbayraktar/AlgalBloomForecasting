import os
import rasterio
import numpy as np
import warnings
import dateutil.parser

from datetime import timedelta
from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib.colors as colors


plt.rc('font', **{'size' : 8})

data_dir = '/.../' # to be filled
subset = 'palmar'  # one of 'bonete', 'baygorria', 'palmar'

threshold1 = 0
threshold2 = 10
threshold3 = 30
threshold4 = 75
# Read data and store in dict with date as key
chlorophyll_data = {}

files = os.listdir(os.path.join(data_dir, subset))
for f in files:
    with rasterio.open(os.path.join(data_dir, subset, f), 'r') as src:
        chlorophyll_data[f.split('_')[1]] = src.read(1)

chlorophyll_data_array = np.asarray(list(chlorophyll_data.values()))
print('chlorophyll data array shape: {}'.format(chlorophyll_data_array.shape))

# Number of temporal samples
print('Number of temporal samples: {}'.format(chlorophyll_data_array.shape[0]))



# Min, max values
print('min, max = {:.3f}, {:.3f}'.format(np.nanmin(chlorophyll_data_array), np.nanmax(chlorophyll_data_array)))

# Number of NaN and non-NaN samples:
samples_nan = np.sum(np.isnan(chlorophyll_data_array))
samples_non_nan = np.sum(~np.isnan(chlorophyll_data_array))
samples_total = np.prod(chlorophyll_data_array.shape)
print('{:d} ({:.3f}%) samples of {:d} total are not NaN.'.format(samples_non_nan, samples_non_nan / samples_total, samples_total))

# Count number of data samples larger than certain threshold
threshold = 150
c = np.sum(chlorophyll_data_array > threshold)
print('{:d} ({:.3f}%) of not-NaN samples larger than threshold {:d}.'.format(c, c / samples_non_nan * 100, threshold))

# Clip values at threshold
chlorophyll_data_array = np.clip(chlorophyll_data_array, a_min=0, a_max=threshold)

figure1,axes1=plt.subplots(1,1)
axes1.set_title('Histogram of chlorophyll a concentration (clipped)')
axes1.set_xlabel('X')
axes1.set_ylabel('Y')
_ = axes1.hist(chlorophyll_data_array.flatten(), bins=100)
figure1.savefig("figure1.png")



flattened = chlorophyll_data_array.flatten()
print('flattened shape: {}'.format(flattened.shape))
flattened[np.where(np.isnan(flattened))] = 0

cond1 = np.logical_and(flattened>threshold1,flattened<threshold2)
cond2 = np.logical_and(flattened>=threshold2,flattened<threshold3)
cond3 = np.logical_and(flattened>=threshold3,flattened<threshold4)
cond4 = flattened[flattened>=threshold4]

print('{:d} ({:.3f}%) of locations have chlorophyll-a concentration larger than threshold {:d}.'.format(np.count_nonzero(cond1), (np.count_nonzero(cond1) / 9411463) * 100, threshold1))
print('{:d} ({:.3f}%) of locations have chlorophyll-a concentration larger than threshold {:d}.'.format(np.count_nonzero(cond2), (np.count_nonzero(cond2) / 9411463) * 100, threshold2))
print('{:d} ({:.3f}%) of locations have chlorophyll-a concentration larger than threshold {:d}.'.format(np.count_nonzero(cond3), (np.count_nonzero(cond3) / 9411463) * 100, threshold3))
print('{:d} ({:.3f}%) of locations have chlorophyll-a concentration larger than threshold {:d}.'.format(np.count_nonzero(cond4), (np.count_nonzero(cond4) / 9411463) * 100, threshold4))



# Plot spatial distribution of non-NaN values

# Count number of measurements per pixel
count_nonnan = np.sum(~np.isnan(chlorophyll_data_array), axis=0)

# Percentage of water-pixels with a meaurement
n = np.count_nonzero(count_nonnan) * chlorophyll_data_array.shape[0]
print('Percentage of pixels with a measurement: {:.3f} %'.format(np.sum(~np.isnan(chlorophyll_data_array)) / n * 100))


# Mean and std of chlorophyll-a concentration per location
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    chlorophyll_mean = np.nanmean(chlorophyll_data_array, axis=0)
    chlorophyll_std = np.nanstd(chlorophyll_data_array, axis=0)

# Replace NaN by zero
chlorophyll_mean[np.where(np.isnan(chlorophyll_mean))] = 0
chlorophyll_std[np.where(np.isnan(chlorophyll_std))] = 0
chlorophyll_mean = np.clip(chlorophyll_mean, a_min=0, a_max=100)
print(chlorophyll_mean.shape)
print(chlorophyll_std.shape)
# Count locations with average chlorophyll-a concentration larger than threshold

figure2, axes2=plt.subplots(1, 1)
axes2.set_title('Spatial distribution of measurements.')
im1=axes2.imshow(count_nonnan,cmap="hot")
figure2.colorbar(im1, orientation='vertical')
figure2.savefig("figure2.png")

figure3, axes3=plt.subplots(1, 1,figsize=(6.4,5.8),dpi=200)
axes3.set_title('Average Chlorophyll-a Concentration',size=10)
im3=axes3.imshow(chlorophyll_mean,cmap=cm.get_cmap('ocean').reversed(),norm=colors.PowerNorm(gamma=0.7))
figure3.colorbar(im3,fraction=0.046, pad=0.04 ,orientation='vertical',ax=axes3)
figure3.savefig("figure3.png")

figurecopy, axescopy=plt.subplots(1, 1,figsize=(6.4,5.8),dpi=100)
axescopy.set_title('Temporal Variation of \n Chlorophyll-a Concentration',size=10)
im4=axescopy.imshow(chlorophyll_std,cmap=cm.get_cmap('cubehelix').reversed())
cb = figurecopy.colorbar(im4, fraction=0.046, pad=0.04,orientation='vertical',ax=axescopy)
cb.ax.set_ylabel('Change in µg/L', rotation=270,labelpad=15)
figurecopy.savefig("figurecopy.png")




dates = sorted(chlorophyll_data.keys())

# # Convert dates to datetime objects, remove time component
dates = [dateutil.parser.parse(d).date() for d in dates]

print("first date:")
print(dates[0])

print("last date:")
print(dates[-1])
# Count number of measurements per day
measurements = {}
for n in range(int((dates[-1] - dates[0]).days) + 1):
    # Count number of measurements on day n
    measurements[dates[0] + timedelta(n)] = np.sum([d == dates[0] + timedelta(n) for d in dates])

# Sanity check: number of measurements should be equal to number of dates
assert np.sum(list(measurements.values())) == len(dates), 'Number of measurements does not match number of dates.'

# Plot number of measurements per day
figure4, axes4=plt.subplots(1,1,figsize=(12, 4))
axes4.set_title('Number of measurements per day')
plt.bar(list(measurements.keys()), list(measurements.values()), width=2.5)
axes4.set_xlabel('Date')
axes4.set_ylabel('Number of measurements')
figure4.tight_layout()
figure4.savefig("figure4.png")
# # Max number of measurements per day
print('Max number of measurements per day: {:d}'.format(np.max(list(measurements.values()))))

# # Percentage of days with more than 0 measurements
print('{:.2f}% of days have at least one measurement.'.format(np.sum(np.array(list(measurements.values())) > 0) / len(measurements) * 100))


# Compute average chlorophyll-a concentration per day
chlorophyll_mean = {}
for i, d in enumerate(chlorophyll_data.keys()):
    # Compute average chlorophyll-a concentration on day d
    chlorophyll_mean[d] = np.nanmean(chlorophyll_data_array[i, :, :])

sorteddict=sorted(chlorophyll_mean.keys())
# Plot average chlorophyll-a concentration per day
figure5, axes5=plt.subplots(1,1,figsize=(12, 4))
plt.title('Average chlorophyll-a concentration per day')
axes5.plot(list(sorteddict), list(chlorophyll_mean.values()))
axes5.set_xlabel('Date')
axes5.set_ylabel('Average chlorophyll-a concentration')
axes5.get_xaxis().set_visible(False)
figure5.tight_layout()
figure5.savefig("figure5.png")