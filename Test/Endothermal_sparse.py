#!/usr/bin/env python3
"""
Temperature Analysis Utility for Sparse Data (Manual Readings)

This script is designed for analyzing temperature data with few data points,
such as manual readings from mercury thermometers. It uses more conservative
fitting methods to avoid overfitting and misleading interpolation, focusing on finding the MINIMUM.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d


# --- load_data, polynomial_model, fit_polynomial, fit_piecewise_linear ไม่เปลี่ยนแปลง ---

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
        if 'Time(s)' not in df.columns or 'Temperature' not in df.columns:
            raise ValueError("CSV must contain 'Time(s)' and 'Temperature' columns")
        
        time = df['Time(s)'].values
        temperature = df['Temperature'].values
        
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
    """Fit a polynomial to the temperature data."""
    time_mean = np.mean(time)
    time_std = np.std(time)
    time_norm = (time - time_mean) / time_std
    
    coeffs = np.polyfit(time_norm, temperature, degree)
    poly_func = np.poly1d(coeffs)
    
    def fitted_func(t):
        t_norm = (t - time_mean) / time_std
        return poly_func(t_norm)
    
    predicted = fitted_func(time)
    residuals = temperature - predicted
    
    return fitted_func, coeffs, residuals

def fit_piecewise_linear(time, temperature):
    """Fit piecewise linear interpolation (connects the dots)."""
    return interp1d(time, temperature, kind='linear',
                    bounds_error=False, fill_value='extrapolate')

# --- ฟังก์ชันหลักในการหาค่าต่ำสุด (แทนที่ find_max_temperature_conservative) ---
def find_min_temperature_conservative(time, temperature, fitted_func):
    """
    Find minimum temperature using a conservative approach.
    
    Returns:
        dict with analysis results for the minimum.
    """
    results = {}

    # Minimum from measured data
    min_idx = np.argmin(temperature)
    results['measured_min_temp'] = temperature[min_idx]
    results['measured_min_time'] = time[min_idx]

    # Minimum from fitted curve (within data range only)
    time_fine = np.linspace(time.min(), time.max(), 1000)
    temp_fine = fitted_func(time_fine)

    min_fit_idx = np.argmin(temp_fine)
    results['fitted_min_temp'] = temp_fine[min_fit_idx]
    results['fitted_min_time'] = time_fine[min_fit_idx]

    # Calculate residuals and uncertainty
    residuals = temperature - fitted_func(time)
    residual_std = np.std(residuals) if len(residuals) > 1 else 0.0
    results['residual_std'] = residual_std

    # Check if fitted min is significantly different from measured
    temp_diff = results['fitted_min_temp'] - results['measured_min_temp']
    results['temp_difference'] = temp_diff # The difference is (Fitted MIN - Measured MIN)
    
    # Check if fitted min is UNREASONABLY LOWER than measured (e.g., artifact)
    if residual_std < 0.01:  # Piecewise case
        time_diff = abs(results['fitted_min_time'] - results['measured_min_time'])
        results['exceeds_uncertainty'] = time_diff > 0.1
        results['uncertainty_reason'] = 'piecewise'
    else: # Polynomial case
        results['exceeds_uncertainty'] = (
            abs(temp_diff) > max(2 * residual_std, 0.5) or # More than 2σ or 0.5°C
            temp_diff < -1.0 # Or more than 1°C LOWER
        )
        results['uncertainty_reason'] = 'statistical'

    return results


# --- ฟังก์ชันการพลอต (แทนที่ plot_results_sparse) ---
def plot_results_min(time, temperature, fitted_func, results,
                         fit_method='polynomial', output_file=None):
    """
    Plot results for sparse data analysis, focusing on the minimum.
    """
    # Create high-resolution points for smooth curve
    time_smooth = np.linspace(time.min(), time.max(), 500)
    temp_smooth = fitted_func(time_smooth)

    # Create the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10),
                                  gridspec_kw={'height_ratios': [3, 1]})

    # Main plot
    ax1.scatter(time, temperature, color='blue', label='Measured Data',
                alpha=0.8, s=100, zorder=4, edgecolors='darkblue', linewidth=1.5)

    # Plot fitted curve
    fit_label = f'{fit_method.capitalize()} Fit'
    ax1.plot(time_smooth, temp_smooth, color='red', label=fit_label,
             linewidth=2.5, zorder=2, alpha=0.7)

    # Mark measured MINIMUM
    meas_min_temp = results['measured_min_temp']
    meas_min_time = results['measured_min_time']

    # Use a circle marker for minimum
    ax1.scatter([meas_min_time], [meas_min_temp],
                color='none', marker='o', s=300,
                edgecolors='green', linewidth=3, zorder=5,
                label=f'Measured Min: {meas_min_temp:.2f}°C at t={meas_min_time:.0f}s')
    ax1.scatter([meas_min_time], [meas_min_temp],
                color='green', marker='v', s=40, zorder=6)

    # Mark fitted minimum if significantly different
    fit_min_temp = results['fitted_min_temp']
    fit_min_time = results['fitted_min_time']

    if results['exceeds_uncertainty']:
        # Fitted min is questionable - show with warning
        ax1.scatter([fit_min_time], [fit_min_temp],
                    color='none', marker='D', s=250, # Diamond marker for uncertain minimum
                    edgecolors='orange', linewidth=2.5, zorder=5, linestyle='--',
                    label=f'Fitted Min: {fit_min_temp:.2f}°C (uncertain!)')
        ax1.scatter([fit_min_time], [fit_min_temp],
                    color='orange', marker='D', s=30, zorder=6, alpha=0.7)

        # Add warning annotation
        y_range = temp_smooth.max() - temp_smooth.min()
        ax1.annotate('⚠ Fitted min may be artifact\nof sparse data',
                     xy=(fit_min_time, fit_min_temp),
                     xytext=(fit_min_time + (time.max()-time.min())*0.15,
                             fit_min_temp - y_range*0.1),
                     ha='left', va='top',
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
    ax1.set_title('Temperature Analysis for Sparse Data (Minimum Focus)',
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


# --- ฟังก์ชัน main (แก้ไขเฉพาะส่วนการเรียกใช้และแสดงผล) ---
def main():
    parser = argparse.ArgumentParser(
        description='Analyze temperature data with few data points (manual readings), focusing on MINIMUM.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s lab_1_Hg.csv
  %(prog)s data.csv --method piecewise

Notes:
  This tool is designed for sparse data (< 15 points) like manual thermometer readings.
  It focuses on the Minimum Temperature, using conservative fitting to avoid misleading artifacts.
        """
    )

    # ... (ส่วน Argument Parsing ไม่เปลี่ยนแปลง) ...
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

    # ... (ส่วน Construct full file path และ Load Data ไม่เปลี่ยนแปลง) ...
    if args.data_dir is None:
        script_dir = Path(__file__).parent.parent
        data_dir = script_dir / 'data'
    else:
        data_dir = Path(args.data_dir)
    filepath = data_dir / args.filename

    print("="*70)
    print("SPARSE DATA TEMPERATURE ANALYSIS (MINIMUM FOCUS)")
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

    # Analyze minimum (MODIFIED FUNCTION CALL)
    print("Analyzing minimum temperature...")
    results = find_min_temperature_conservative(time, temperature, fitted_func)

    # Display results
    print("\n" + "="*70)
    print("ANALYSIS RESULTS: MINIMUM TEMPERATURE")
    print("="*70)

    print("\n1. MEASURED MINIMUM (from actual data points):")
    print(f"   Temperature: {results['measured_min_temp']:.4f}°C")
    print(f"   Time:        {results['measured_min_time']:.2f}s")

    print(f"\n2. FITTED MINIMUM (from {fit_method}):")
    print(f"   Temperature: {results['fitted_min_temp']:.4f}°C")
    print(f"   Time:        {results['fitted_min_time']:.2f}s")

    print(f"\n3. FIT QUALITY:")
    print(f"   Residual Std Dev:  {results['residual_std']:.4f}°C")
    print(f"   Temperature Diff (Fitted-Measured):  {results['temp_difference']:.4f}°C")

    print(f"\n4. RECOMMENDATION:")
    if results['exceeds_uncertainty']:
        print("   ⚠ CAUTION: Fitted minimum differs from measured data!")
        print(f"   The fitted min ({results['fitted_min_temp']:.2f}°C)")
        print(f"   vs measured min ({results['measured_min_temp']:.2f}°C)")
        print(f"   Difference: {results['temp_difference']:.2f}°C")
        print("\n   SUGGESTED INTERPRETATION:")
        print(f"   → Report MEASURED minimum: {results['measured_min_temp']:.2f}°C at {results['measured_min_time']:.0f}s")
        print("   → The fitted curve may show interpolation artifacts")
        print("   → For sparse data, trust the direct measurements")
    else:
        print("   ✓ Fitted minimum matches measured data")
        print(f"   → Report measured min: {results['measured_min_temp']:.2f}°C at {results['measured_min_time']:.0f}s")
        print(f"   → Fitted curve confirms: {results['fitted_min_temp']:.2f}°C at {results['fitted_min_time']:.2f}s")

    print("="*70)

    # Plot results (MODIFIED FUNCTION CALL)
    print("\nGenerating plot...")
    plot_results_min(time, temperature, fitted_func, results,
                             fit_method=fit_method, output_file=args.output)

    if not args.output:
        print("\nClose the plot window to exit.")


if __name__ == '__main__':
    main()