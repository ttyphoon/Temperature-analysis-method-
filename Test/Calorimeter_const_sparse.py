#!/usr/bin/env python3
"""
Temperature Analysis Utility for Sparse Data (Manual Readings)

This script is designed for analyzing temperature data with few data points,
such as manual readings from mercury thermometers. It uses more conservative
fitting methods to avoid overfitting and misleading interpolation.

Key differences from analyze_temperature.py:
1. Uses polynomial fitting instead of spline for better behavior with sparse data
2. Provides uncertainty estimates
3. Shows both the fit and the measured maximum
4. More conservative in claiming maxima between points
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d


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


def polynomial_model(x, *coeffs):
    """Polynomial model for fitting."""
    return sum(c * x**i for i, c in enumerate(coeffs))


def fit_polynomial(time, temperature, degree=3):
    """
    Fit a polynomial to the temperature data.

    Args:
        time: Time values
        temperature: Temperature values
        degree: Polynomial degree (default: 3 for cubic)

    Returns:
        tuple: (fitted_function, coefficients, residuals)
    """
    # Normalize time to improve numerical stability
    time_mean = np.mean(time)
    time_std = np.std(time)
    time_norm = (time - time_mean) / time_std

    # Fit polynomial
    coeffs = np.polyfit(time_norm, temperature, degree)
    poly_func = np.poly1d(coeffs)

    # Create function that works with original time scale
    def fitted_func(t):
        t_norm = (t - time_mean) / time_std
        return poly_func(t_norm)

    # Calculate residuals
    predicted = fitted_func(time)
    residuals = temperature - predicted

    return fitted_func, coeffs, residuals


def fit_piecewise_linear(time, temperature):
    """
    Fit piecewise linear interpolation (connects the dots).
    More conservative - doesn't extrapolate beyond data.
    """
    return interp1d(time, temperature, kind='linear',
                    bounds_error=False, fill_value='extrapolate')


def find_max_temperature_conservative(time, temperature, fitted_func, method='measured'):
    """
    Find maximum temperature using a conservative approach.

    Args:
        time: Original time data
        temperature: Original temperature data
        fitted_func: Fitted function
        method: 'measured' (only from data), 'fitted' (from fit), or 'both'

    Returns:
        dict with analysis results
    """
    results = {}

    # Maximum from measured data
    max_idx = np.argmax(temperature)
    results['measured_max_temp'] = temperature[max_idx]
    results['measured_max_time'] = time[max_idx]

    # Maximum from fitted curve (within data range only)
    time_fine = np.linspace(time.min(), time.max(), 1000)
    temp_fine = fitted_func(time_fine)

    max_fit_idx = np.argmax(temp_fine)
    results['fitted_max_temp'] = temp_fine[max_fit_idx]
    results['fitted_max_time'] = time_fine[max_fit_idx]

    # Calculate if fitted max is significantly different from measured
    residuals = temperature - fitted_func(time)
    residual_std = np.std(residuals) if len(residuals) > 1 else 0.0
    results['residual_std'] = residual_std

    # Check if fitted max exceeds measured data by more than expected noise
    temp_diff = results['fitted_max_temp'] - results['measured_max_temp']
    results['temp_difference'] = temp_diff

    # For piecewise linear, residuals are zero, so use a different criterion
    # Check if fitted max is at a different location than measured max
    if residual_std < 0.01:  # Essentially perfect fit (piecewise)
        # Check if max is at same location
        time_diff = abs(results['fitted_max_time'] - results['measured_max_time'])
        results['exceeds_uncertainty'] = time_diff > 0.1  # Different by more than 0.1s
        results['uncertainty_reason'] = 'piecewise'
    else:
        # Check if temp difference exceeds 2 standard deviations
        # But also check if fitted max is unreasonably higher
        results['exceeds_uncertainty'] = (
            abs(temp_diff) > max(2 * residual_std, 0.5) or  # More than 2σ or 0.5°C
            temp_diff > 1.0  # Or more than 1°C higher
        )
        results['uncertainty_reason'] = 'statistical'

    return results


def plot_results_sparse(time, temperature, fitted_func, results,
                       fit_method='polynomial', output_file=None):
    """
    Plot results for sparse data analysis.
    """
    # Create high-resolution points for smooth curve
    time_smooth = np.linspace(time.min(), time.max(), 500)
    temp_smooth = fitted_func(time_smooth)

    # Create the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10),
                                    gridspec_kw={'height_ratios': [3, 1]})

    # Main plot
    # Plot original data points - larger for sparse data
    ax1.scatter(time, temperature, color='blue', label='Measured Data',
                alpha=0.8, s=100, zorder=4, edgecolors='darkblue', linewidth=1.5)

    # Plot fitted curve
    fit_label = f'{fit_method.capitalize()} Fit'
    ax1.plot(time_smooth, temp_smooth, color='red', label=fit_label,
             linewidth=2.5, zorder=2, alpha=0.7)

    # Mark measured maximum
    meas_max_temp = results['measured_max_temp']
    meas_max_time = results['measured_max_time']

    ax1.scatter([meas_max_time], [meas_max_temp],
                color='none', marker='s', s=300,
                edgecolors='darkgreen', linewidth=3, zorder=5,
                label=f'Measured Max: {meas_max_temp:.2f}°C at t={meas_max_time:.0f}s')
    ax1.scatter([meas_max_time], [meas_max_temp],
                color='darkgreen', marker='s', s=40, zorder=6)

    # Mark fitted maximum if significantly different
    fit_max_temp = results['fitted_max_temp']
    fit_max_time = results['fitted_max_time']

    if results['exceeds_uncertainty']:
        # Fitted max is questionable - show with warning
        ax1.scatter([fit_max_time], [fit_max_temp],
                    color='none', marker='o', s=250,
                    edgecolors='orange', linewidth=2.5, zorder=5, linestyle='--',
                    label=f'Fitted Max: {fit_max_temp:.2f}°C (uncertain!)')
        ax1.scatter([fit_max_time], [fit_max_temp],
                    color='orange', marker='o', s=30, zorder=6, alpha=0.7)

        # Add warning annotation
        y_range = temp_smooth.max() - temp_smooth.min()
        ax1.annotate('⚠ Fitted max may be artifact\nof sparse data',
                    xy=(fit_max_time, fit_max_temp),
                    xytext=(fit_max_time + (time.max()-time.min())*0.15,
                           fit_max_temp + y_range*0.1),
                    ha='left', va='bottom',
                    fontsize=10, color='orange', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow',
                             edgecolor='orange', alpha=0.8),
                    arrowprops=dict(arrowstyle='->', color='orange', lw=2))

    # Add gridlines for the measured points
    ax1.vlines(time, temperature.min(), temperature,
               colors='gray', linestyles=':', alpha=0.3, linewidth=1)

    # Labels and formatting
    ax1.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Temperature (°C)', fontsize=12, fontweight='bold')
    ax1.set_title('Temperature Analysis for Sparse Data (Manual Readings)',
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10, framealpha=0.9)
    ax1.grid(True, alpha=0.3)

    # Residual plot
    residuals = temperature - fitted_func(time)
    ax2.scatter(time, residuals, color='purple', s=80, alpha=0.7, zorder=3)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.5)
    ax2.axhline(y=2*results['residual_std'], color='red',
                linestyle='--', linewidth=1, alpha=0.5,
                label=f'±2σ = ±{2*results["residual_std"]:.2f}°C')
    ax2.axhline(y=-2*results['residual_std'], color='red',
                linestyle='--', linewidth=1, alpha=0.5)

    ax2.set_xlabel('Time (s)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Residual (°C)', fontsize=11, fontweight='bold')
    ax2.set_title('Fit Quality: Residuals (Measured - Fitted)', fontsize=12)
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save or show
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {output_file}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Analyze temperature data with few data points (manual readings)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s lab_1_Hg.csv
  %(prog)s lab_1_Hg.csv --degree 2
  %(prog)s manual_data.csv -o output.png
  %(prog)s data.csv --method piecewise

Notes:
  This tool is designed for sparse data (< 15 points) like manual thermometer readings.
  It uses conservative fitting to avoid misleading interpolation artifacts.
        """
    )

    parser.add_argument('filename', type=str,
                        help='CSV file containing temperature data')
    parser.add_argument('--degree', type=int, default=3,
                        help='Polynomial degree for fitting (default: 3, only used with --method polynomial)')
    parser.add_argument('--method', type=str, default='piecewise',
                        choices=['polynomial', 'piecewise'],
                        help='Fitting method (default: piecewise - recommended for sparse data)')
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

    print("="*70)
    print("SPARSE DATA TEMPERATURE ANALYSIS")
    print("For Manual Readings (Mercury Thermometer, etc.)")
    print("="*70)
    print(f"\nLoading data from: {filepath}")

    # Load data
    time, temperature = load_data(filepath)
    print(f"Loaded {len(time)} data points")

    if len(time) >= 15:
        print("\n⚠ WARNING: You have ≥15 data points.")
        print("Consider using 'analyze_temperature.py' for denser data.")

    print(f"Time range: {time.min():.2f}s to {time.max():.2f}s")
    print(f"Temperature range: {temperature.min():.2f}°C to {temperature.max():.2f}°C")

    # Fit data
    print(f"\nFitting {args.method} model...")

    if args.method == 'polynomial':
        fitted_func, coeffs, residuals = fit_polynomial(time, temperature, args.degree)
        fit_method = f'polynomial (degree {args.degree})'
    else:  # piecewise
        fitted_func = fit_piecewise_linear(time, temperature)
        residuals = temperature - fitted_func(time)
        fit_method = 'piecewise linear'

    # Analyze maximum
    print("Analyzing maximum temperature...")
    results = find_max_temperature_conservative(time, temperature, fitted_func)

    # Display results
    print("\n" + "="*70)
    print("ANALYSIS RESULTS")
    print("="*70)

    print("\n1. MEASURED MAXIMUM (from actual data points):")
    print(f"   Temperature: {results['measured_max_temp']:.4f}°C")
    print(f"   Time:        {results['measured_max_time']:.2f}s")

    print(f"\n2. FITTED MAXIMUM (from {fit_method}):")
    print(f"   Temperature: {results['fitted_max_temp']:.4f}°C")
    print(f"   Time:        {results['fitted_max_time']:.2f}s")

    print(f"\n3. FIT QUALITY:")
    print(f"   Residual Std Dev:  {results['residual_std']:.4f}°C")
    print(f"   Temperature Diff:  {results['temp_difference']:.4f}°C")

    print(f"\n4. RECOMMENDATION:")
    if results['exceeds_uncertainty']:
        print("   ⚠ CAUTION: Fitted maximum differs from measured data!")
        print(f"   The fitted max ({results['fitted_max_temp']:.2f}°C)")
        print(f"   vs measured max ({results['measured_max_temp']:.2f}°C)")
        print(f"   Difference: {results['temp_difference']:.2f}°C")
        print("\n   SUGGESTED INTERPRETATION:")
        print(f"   → Report MEASURED maximum: {results['measured_max_temp']:.2f}°C at {results['measured_max_time']:.0f}s")
        print("   → The fitted curve may show interpolation artifacts")
        print("   → For sparse data, trust the direct measurements")
    else:
        print("   ✓ Fitted maximum matches measured data")
        print(f"   → Report measured max: {results['measured_max_temp']:.2f}°C at {results['measured_max_time']:.0f}s")
        print(f"   → Fitted curve confirms: {results['fitted_max_temp']:.2f}°C at {results['fitted_max_time']:.2f}s")

    print("="*70)

    # Plot results
    print("\nGenerating plot...")
    plot_results_sparse(time, temperature, fitted_func, results,
                       fit_method=fit_method, output_file=args.output)

    if not args.output:
        print("\nClose the plot window to exit.")


if __name__ == '__main__':
    main()
