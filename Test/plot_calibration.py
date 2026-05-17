import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
df = pd.read_csv('Error curve.csv', sep='\t', header=0)

# Extract relevant columns
set_point = df['Set point']
time_col = df['Time (s) min:sec']
tref = df['Tref ( ºC )']
tds = df['TDS ( ºC )']
error = df['Difference ( ºC )']

print("Error values (Difference) with 2 decimal places:")
for i, err in enumerate(error):
    print(f"Set point {set_point.iloc[i]}: {err:.2f} °C")

print(df.head())

# Create subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

# Plot Tref and TDS vs Time
ax1.plot(time_col, tref, 'b-o', label='Reference Temperature (Tref)')
ax1.plot(time_col, tds, 'r-s', label='Measured Temperature (TDS)')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Temperature (°C)')
ax1.set_title('Temperature vs Time')
ax1.grid(True)
ax1.legend()

# Plot error vs Time
ax2.plot(time_col, error, 'g-^', label='Error (Difference)')
# Linear fit for error trend
coeff_error = np.polyfit(time_col, error, 1)
poly_error = np.poly1d(coeff_error)
ax2.plot(time_col, poly_error(time_col), 'm--', label=f'Error Trend: y = {coeff_error[0]:.6f}x + {coeff_error[1]:.4f}')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Error (°C)')
ax2.set_title('Calibration Error vs Time with Trend')
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.savefig('calibration_analysis.png')
print("Calibration analysis plot saved as calibration_analysis.png")