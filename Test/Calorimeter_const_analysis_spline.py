#!/usr/bin/env python3
"""
Temperature Analysis Utility

This script analyzes temperature vs time data by:
1. Loading CSV data with time and temperature columns
2. Fitting a spline to the data
3. Finding the maximum temperature using the first derivative
4. Plotting the results
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize_scalar


def load_data(filepath):
    """
    Load CSV data from file.

    Args:
        filepath: Path to CSV file with Time(s) and Temperature columns

    Returns:
        tuple: (time_array, temperature_array)
    """
    try:
        df = pd.read_csv(filepath)

        # Check for expected columns
        if 'Time(s)' not in df.columns or 'Temperature' not in df.columns:
            raise ValueError("CSV must contain 'Time(s)' and 'Temperature' columns")

        time = df['Time(s)'].values
        temperature = df['Temperature'].values

        # Remove any NaN values
        mask = ~(np.isnan(time) | np.isnan(temperature))
        time = time[mask]
        temperature = temperature[mask]

        if len(time) == 0:
            raise ValueError("No valid data points found")

        return time, temperature

    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)


def fit_spline(time, temperature, smoothing=None):
    """
    Fit a univariate spline to the temperature data.

    Args:
        time: Time values
        temperature: Temperature values
        smoothing: Smoothing parameter (None for very tight fit, 0 for interpolation)

    Returns:
        UnivariateSpline object
    """
    # Use UnivariateSpline with appropriate smoothing
    # If smoothing is None or 0, use a very small value for tight fit
    # This makes the spline pass very close to all data points
    if smoothing is None:
        # Use a very small smoothing factor - this makes the curve follow data closely
        # but still maintains smoothness for derivative calculation
        smoothing = 0.0

    spline = UnivariateSpline(time, temperature, s=smoothing, k=3)
    return spline


def find_max_temperature(spline, time_range):
    """
    Find the maximum temperature using the first derivative of the spline.

    This function finds the global maximum by:
    1. Evaluating the spline on a fine grid to get the global maximum
    2. Using optimization to refine the location
    3. Checking boundary points as well

    Args:
        spline: UnivariateSpline object
        time_range: Tuple of (min_time, max_time)

    Returns:
        tuple: (time_at_max, max_temperature)
    """
    # Evaluate spline on a fine grid to find approximate global maximum
    time_fine = np.linspace(time_range[0], time_range[1], 1000)
    temp_fine = spline(time_fine)

    # Find the index of maximum temperature on the grid
    max_idx = np.argmax(temp_fine)
    approx_max_time = time_fine[max_idx]
    approx_max_temp = temp_fine[max_idx]

    # Refine the maximum using optimization in a neighborhood around the grid maximum
    # Use a window around the approximate maximum
    window_size = (time_range[1] - time_range[0]) * 0.1  # 10% window
    search_min = max(time_range[0], approx_max_time - window_size)
    search_max = min(time_range[1], approx_max_time + window_size)

    # Optimize to find precise maximum
    def neg_temp(t):
        return -spline(t)

    result = minimize_scalar(neg_temp, bounds=(search_min, search_max), method='bounded')

    time_at_max = result.x
    max_temp = spline(time_at_max)

    # Double-check against boundary points
    temp_at_start = spline(time_range[0])
    temp_at_end = spline(time_range[1])

    if temp_at_start > max_temp:
        time_at_max = time_range[0]
        max_temp = temp_at_start
    if temp_at_end > max_temp:
        time_at_max = time_range[1]
        max_temp = temp_at_end

    return time_at_max, max_temp


def plot_results(time, temperature, spline, time_at_max, max_temp, smoothing_param=0.0, output_file=None):
    """
    Plot the original data, spline fit, and maximum temperature point.

    Args:
        time: Original time data
        temperature: Original temperature data
        spline: Fitted spline
        time_at_max: Time at maximum temperature
        max_temp: Maximum temperature value
        output_file: Optional file path to save the plot
    """
    # Create high-resolution time points for smooth spline curve
    time_smooth = np.linspace(time.min(), time.max(), 500)
    temp_smooth = spline(time_smooth)

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot original data points
    plt.scatter(time, temperature, color='blue', label='Data Points',
                alpha=0.6, s=50, zorder=3)

    # Plot spline fit with smoothing parameter in label
    spline_label = f'Spline Fit (s={smoothing_param:.1f})'
    plt.plot(time_smooth, temp_smooth, color='red', label=spline_label,
             linewidth=2, zorder=2)

    # Plot maximum temperature point - using a smaller marker with hollow center
    plt.scatter([time_at_max], [max_temp], color='none', marker='o',
                s=200, edgecolors='green', linewidth=2.5, zorder=4,
                label=f'Max: {max_temp:.2f}°C at t={time_at_max:.2f}s')

    # Add a small filled point at the center
    plt.scatter([time_at_max], [max_temp], color='green', marker='o',
                s=30, zorder=5)

    # Add vertical line at maximum
    plt.axvline(x=time_at_max, color='green', linestyle='--',
                alpha=0.4, linewidth=1.5, zorder=1)

    # Add horizontal line at maximum temperature
    plt.axhline(y=max_temp, color='green', linestyle=':',
                alpha=0.3, linewidth=1, zorder=1)

    # Add annotation with arrow pointing to maximum
    y_range = temp_smooth.max() - temp_smooth.min()
    annotation_offset = y_range * 0.15  # Position annotation above the point

    plt.annotate(f'{max_temp:.2f}°C',
                xy=(time_at_max, max_temp),
                xytext=(time_at_max, max_temp + annotation_offset),
                ha='center', va='bottom',
                fontsize=11, fontweight='bold', color='green',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                         edgecolor='green', alpha=0.8),
                arrowprops=dict(arrowstyle='->', color='green', lw=1.5),
                zorder=6)

    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Temperature (°C)', fontsize=12)
    plt.title('Temperature vs Time Analysis', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)

    # Add analysis parameters as text box
    n_points = len(time)
    params_text = (
        f'Analysis Parameters:\n'
        f'Data points: {n_points}\n'
        f'Smoothing (s): {smoothing_param:.2f}\n'
        f'Spline order: cubic (k=3)'
    )

    # Position the text box in the lower right corner
    plt.text(0.98, 0.02, params_text,
             transform=plt.gca().transAxes,
             fontsize=9,
             verticalalignment='bottom',
             horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.5',
                      facecolor='wheat',
                      edgecolor='gray',
                      alpha=0.9))

    # Adjust layout
    plt.tight_layout()

    # Save or show
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {output_file}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Analyze temperature data using spline fitting',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s lab_1_Hg.csv
  %(prog)s lab_2_Thermab.csv -s 0.5
  %(prog)s data.csv -o output.png
        """
    )

    parser.add_argument('filename', type=str,
                        help='CSV file containing temperature data')
    parser.add_argument('-s', '--smoothing', type=float, default=None,
                        help='Smoothing parameter for spline fit (default: 0 for tight fit, higher values for more smoothing)')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output file for plot (default: display on screen)')
    parser.add_argument('-d', '--data-dir', type=str,
                        default=None,
                        help='Directory containing data files (default: ./data)')

    args = parser.parse_args()

    # Construct full file path
    # If no data-dir specified, use ./data relative to script location
    if args.data_dir is None:
        script_dir = Path(__file__).parent.parent  # Go up to project root from src/
        data_dir = script_dir / 'data'
    else:
        data_dir = Path(args.data_dir)
    filepath = data_dir / args.filename

    print(f"Loading data from: {filepath}")

    # Load data
    time, temperature = load_data(filepath)
    print(f"Loaded {len(time)} data points")
    print(f"Time range: {time.min():.2f}s to {time.max():.2f}s")
    print(f"Temperature range: {temperature.min():.2f}°C to {temperature.max():.2f}°C")

    # Fit spline
    smoothing_value = args.smoothing if args.smoothing is not None else 0.0
    print(f"\nFitting spline (smoothing={smoothing_value})...")
    spline = fit_spline(time, temperature, smoothing=args.smoothing)

    # Find maximum temperature
    print("Finding maximum temperature from derivative...")
    time_at_max, max_temp = find_max_temperature(spline, (time.min(), time.max()))

    # Display results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Maximum Temperature: {max_temp:.4f}°C")
    print(f"Time at Maximum:     {time_at_max:.4f}s")
    print("="*60)

    # Plot results
    print("\nGenerating plot...")
    plot_results(time, temperature, spline, time_at_max, max_temp,
                smoothing_param=smoothing_value, output_file=args.output)

    if not args.output:
        print("\nClose the plot window to exit.")


if __name__ == '__main__':
    main()