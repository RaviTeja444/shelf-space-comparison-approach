# visualizer.py
"""Create publication-quality visualizations for research paper"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

# Set publication-quality defaults
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 12
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['lines.linewidth'] = 1.5

# Academic color palette
COLORS = {
    'Optimized DP': '#1f77b4',      # Blue
    'Greedy': '#2ca02c',            # Green  
    'Simulated Annealing': '#ff7f0e', # Orange
    'Pure LLM': '#d62728',           # Red
    'Hybrid LLM-DP': '#9467bd'       # Purple
}

def plot_profit_comparison(results, save_path="profit_comparison.png"):
    """Create publication-quality profit comparison figure"""
    methods = [r.method for r in results]
    profits = [r.total_profit for r in results]
    
    # Create figure with tight layout
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    
    # Create bars with academic colors
    bars = ax.bar(methods, profits, 
                   color=[COLORS.get(m, '#333333') for m in methods],
                   edgecolor='black', linewidth=0.5, alpha=0.85)
    
    # Add value labels on bars
    for bar, profit in zip(bars, profits):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max(profits)*0.01,
                f'${profit:.0f}', ha='center', va='bottom', fontsize=9)
    
    # Formatting
    ax.set_xlabel('Algorithm', fontweight='bold')
    ax.set_ylabel('Total Profit ($)', fontweight='bold')
    ax.set_title('(a) Total Profit Comparison Across All Methods', pad=10)
    
    # Grid
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {save_path} - Publication-quality profit comparison")

def plot_runtime_comparison(results, save_path="runtime_comparison.png"):
    """Create publication-quality runtime comparison figure"""
    methods = [r.method for r in results]
    times = [r.computation_time * 1000 for r in results]  # Convert to ms
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    
    # Create bars
    bars = ax.bar(methods, times,
                   color=[COLORS.get(m, '#333333') for m in methods],
                   edgecolor='black', linewidth=0.5, alpha=0.85)
    
    # Add value labels
    for bar, time in zip(bars, times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height * 1.05,
                f'{time:.1f}', ha='center', va='bottom', fontsize=9)
    
    # Formatting
    ax.set_xlabel('Algorithm', fontweight='bold')
    ax.set_ylabel('Computation Time (ms)', fontweight='bold')
    ax.set_title('(b) Computational Time Comparison', pad=10)
    ax.set_yscale('log')
    
    # Grid
    ax.grid(True, axis='y', alpha=0.3, linestyle='--', which='both')
    ax.set_axisbelow(True)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {save_path} - Publication-quality runtime comparison")

def plot_utilization_comparison(results, capacity, save_path="utilization_comparison.png"):
    """Create publication-quality shelf utilization figure"""
    methods = [r.method for r in results]
    utilizations = [r.get_utilization(capacity) for r in results]
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    
    # Create bars
    bars = ax.bar(methods, utilizations,
                   color=[COLORS.get(m, '#333333') for m in methods],
                   edgecolor='black', linewidth=0.5, alpha=0.85)
    
    # Add value labels
    for bar, util in zip(bars, utilizations):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{util:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # Add reference line at 100%
    ax.axhline(y=100, color='black', linestyle='--', alpha=0.5, linewidth=0.8)
    ax.text(0.02, 0.95, '100% Capacity', transform=ax.transAxes, 
            fontsize=9, alpha=0.7)
    
    # Formatting
    ax.set_xlabel('Algorithm', fontweight='bold')
    ax.set_ylabel('Shelf Utilization (%)', fontweight='bold')
    ax.set_title('(c) Shelf Space Utilization Comparison', pad=10)
    ax.set_ylim(0, 110)
    
    # Grid
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {save_path} - Publication-quality utilization comparison")

def plot_dp_vs_llm_comparison(results, save_path="dp_vs_llm_comparison.png"):
    """Create focused comparison figure for DP vs LLM approaches"""
    # Filter for specific methods
    dp_result = next((r for r in results if r.method == "Optimized DP"), None)

    hybrid_result = next((r for r in results if r.method == "Hybrid LLM-DP"), None)
    
    filtered_results = [r for r in [dp_result, hybrid_result] if r]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Data preparation
    methods = [r.method for r in filtered_results]
    profits = [r.total_profit for r in filtered_results]
    times = [r.computation_time * 1000 for r in filtered_results]
    
    # Subplot 1: Profit comparison
    bars1 = ax1.bar(methods, profits,
                     color=[COLORS.get(m, '#333333') for m in methods],
                     edgecolor='black', linewidth=0.5, alpha=0.85)
    
    # Add optimal line
    optimal_profit = dp_result.total_profit
    ax1.axhline(y=optimal_profit, color='black', linestyle=':', alpha=0.5, linewidth=1)
    ax1.text(0.02, 0.95, 'Optimal', transform=ax1.transAxes, fontsize=9, alpha=0.7)
    
    # Add value labels
    for bar, profit in zip(bars1, profits):
        height = bar.get_height()
        percentage = (profit / optimal_profit) * 100
        ax1.text(bar.get_x() + bar.get_width()/2., height + optimal_profit*0.01,
                f'${profit:.0f}\n({percentage:.1f}%)', 
                ha='center', va='bottom', fontsize=9)
    
    ax1.set_ylabel('Total Profit ($)', fontweight='bold')
    ax1.set_title('(d) Profit Achievement', pad=10)
    ax1.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax1.set_axisbelow(True)
    
    # Subplot 2: Efficiency (profit per ms)
    efficiencies = [p/t for p, t in zip(profits, times)]
    bars2 = ax2.bar(methods, efficiencies,
                     color=[COLORS.get(m, '#333333') for m in methods],
                     edgecolor='black', linewidth=0.5, alpha=0.85)
    
    # Add value labels
    for bar, eff in zip(bars2, efficiencies):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height * 1.02,
                f'{eff:.2f}', ha='center', va='bottom', fontsize=9)
    
    ax2.set_ylabel('Profit per Millisecond ($/ms)', fontweight='bold')
    ax2.set_title('(e) Computational Efficiency', pad=10)
    ax2.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax2.set_axisbelow(True)
    
    # Remove top and right spines for both subplots
    for ax in [ax1, ax2]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # Adjust x-axis labels
    for ax in [ax1, ax2]:
        ax.set_xticklabels(['DP', 'Hybrid'], rotation=0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {save_path} - Publication-quality DP vs LLM comparison")

def plot_scalability_analysis(all_experiments, save_path="scalability_analysis.png"):
    """Create scalability analysis figure"""
    if len(all_experiments) < 2:
        print("Need at least 2 experiments for scalability analysis")
        return
    
    # Extract data
    methods = ["Optimized DP", "Hybrid LLM-DP"]
    method_data = {m: {'sizes': [], 'times': [], 'profits': []} for m in methods}
    
    for results, products, capacity in all_experiments:
        size = len(products)
        for r in results:
            if r.method in methods:
                method_data[r.method]['sizes'].append(size)
                method_data[r.method]['times'].append(r.computation_time * 1000)
                method_data[r.method]['profits'].append(r.total_profit)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Plot 1: Time scalability
    for method in methods:
        if method_data[method]['sizes']:
            ax1.plot(method_data[method]['sizes'], 
                    method_data[method]['times'],
                    marker='o', markersize=6,
                    label=method.replace(' LLM-DP', ''),
                    color=COLORS.get(method, '#333333'),
                    linewidth=1.5, alpha=0.85)
    
    ax1.set_xlabel('Number of Products', fontweight='bold')
    ax1.set_ylabel('Computation Time (ms)', fontweight='bold')
    ax1.set_title('(f) Computational Scalability', pad=10)
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(frameon=True, fancybox=False, shadow=False, framealpha=0.9)
    ax1.set_axisbelow(True)
    
    # Plot 2: Profit scalability
    for method in methods:
        if method_data[method]['sizes']:
            ax2.plot(method_data[method]['sizes'], 
                    method_data[method]['profits'],
                    marker='o', markersize=6,
                    label=method.replace(' LLM-DP', ''),
                    color=COLORS.get(method, '#333333'),
                    linewidth=1.5, alpha=0.85)
    
    ax2.set_xlabel('Number of Products', fontweight='bold')
    ax2.set_ylabel('Total Profit ($)', fontweight='bold')
    ax2.set_title('(g) Profit Scalability', pad=10)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(frameon=True, fancybox=False, shadow=False, framealpha=0.9)
    ax2.set_axisbelow(True)
    
    # Remove top and right spines
    for ax in [ax1, ax2]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {save_path} - Publication-quality scalability analysis")

def save_results_csv(results, products, capacity, filename="results_summary.csv"):
    """Save results to CSV file"""
    data = []
    for r in results:
        data.append({
            'Method': r.method,
            'Total Profit': round(r.total_profit, 2),
            'Total Width': r.total_width,
            'Utilization %': round(r.get_utilization(capacity), 1),
            'Computation Time (ms)': round(r.computation_time * 1000, 2),
            'Products Allocated': len(r.allocations)
        })
    
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")

def print_summary(results, capacity):
    """Print summary of results"""
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    # Find optimal profit
    dp_result = next((r for r in results if r.method == "Optimized DP"), None)
    optimal_profit = dp_result.total_profit if dp_result else max(r.total_profit for r in results)
    
    for r in results:
        optimality = (r.total_profit / optimal_profit) * 100
        print(f"\n{r.method}:")
        print(f"  Total Profit: ${r.total_profit:.2f} ({optimality:.1f}% of optimal)")
        print(f"  Space Used: {r.total_width}/{capacity} cm ({r.get_utilization(capacity):.1f}%)")
        print(f"  Time: {r.computation_time*1000:.2f} ms")
        print(f"  Products Allocated: {len(r.allocations)}")
    
    # Find best
    best_profit = max(results, key=lambda x: x.total_profit)
    fastest = min(results, key=lambda x: x.computation_time)
    
    print("\n" + "-"*40)
    print(f"Best Profit: {best_profit.method} (${best_profit.total_profit:.2f})")
    print(f"Fastest: {fastest.method} ({fastest.computation_time*1000:.2f} ms)")
    print("-"*40)# visualizer.py
"""Create visualizations for results"""

import matplotlib.pyplot as plt
import pandas as pd

def plot_profit_comparison(results, save_path="profit_comparison.png"):
    """Bar chart comparing profits across methods"""
    methods = [r.method for r in results]
    profits = [r.total_profit for r in results]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(methods, profits, color=['blue', 'green', 'red', 'orange', 'purple'])
    
    # Add value labels on bars
    for bar, profit in zip(bars, profits):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'${profit:.2f}', ha='center', va='bottom')
    
    plt.title('Total Profit Comparison', fontsize=16)
    plt.xlabel('Method', fontsize=12)
    plt.ylabel('Total Profit ($)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_runtime_comparison(results, save_path="runtime_comparison.png"):
    """Bar chart comparing runtimes"""
    methods = [r.method for r in results]
    times = [r.computation_time * 1000 for r in results]  # Convert to ms
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(methods, times, color=['blue', 'green', 'red', 'orange', 'purple'])
    
    # Add value labels
    for bar, time in zip(bars, times):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{time:.2f}ms', ha='center', va='bottom')
    
    plt.title('Computation Time Comparison', fontsize=16)
    plt.xlabel('Method', fontsize=12)
    plt.ylabel('Time (milliseconds)', fontsize=12)
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_utilization_comparison(results, capacity, save_path="utilization_comparison.png"):
    """Bar chart comparing shelf utilization"""
    methods = [r.method for r in results]
    utilizations = [r.get_utilization(capacity) for r in results]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(methods, utilizations, color=['blue', 'green', 'red', 'orange', 'purple'])
    
    # Add value labels
    for bar, util in zip(bars, utilizations):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{util:.1f}%', ha='center', va='bottom')
    
    plt.title('Shelf Space Utilization', fontsize=16)
    plt.xlabel('Method', fontsize=12)
    plt.ylabel('Utilization (%)', fontsize=12)
    plt.ylim(0, 110)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def save_results_csv(results, products, capacity, filename="results_summary.csv"):
    """Save results to CSV file"""
    data = []
    for r in results:
        data.append({
            'Method': r.method,
            'Total Profit': r.total_profit,
            'Total Width': r.total_width,
            'Utilization %': r.get_utilization(capacity),
            'Computation Time (ms)': r.computation_time * 1000,
            'Products Allocated': len(r.allocations)
        })
    
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")

def print_summary(results, capacity):
    """Print summary of results"""
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    for r in results:
        print(f"\n{r.method}:")
        print(f"  Total Profit: ${r.total_profit:.2f}")
        print(f"  Space Used: {r.total_width}/{capacity} cm ({r.get_utilization(capacity):.1f}%)")
        print(f"  Time: {r.computation_time*1000:.2f} ms")
        print(f"  Products Allocated: {len(r.allocations)}")
    
    # Find best
    best_profit = max(results, key=lambda x: x.total_profit)
    fastest = min(results, key=lambda x: x.computation_time)
    
    print("\n" + "-"*40)
    print(f"Best Profit: {best_profit.method} (${best_profit.total_profit:.2f})")
    print(f"Fastest: {fastest.method} ({fastest.computation_time*1000:.2f} ms)")
    print("-"*40)