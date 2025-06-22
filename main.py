# main.py
"""Main script to run all experiments"""

import os
from data_generator import generate_products
from optimized_dp import solve_dp, find_gcd
from pure_llm import solve_pure_llm
from hybrid_llm_dp import solve_hybrid_llm_dp
from visualizer import (plot_profit_comparison, plot_runtime_comparison, 
                       plot_utilization_comparison, plot_dp_vs_llm_comparison,
                       plot_scalability_analysis, save_results_csv, print_summary)

def analyze_products(products):
    """Analyze product characteristics"""
    print("\nProduct Analysis:")
    widths = [p.width for p in products]
    categories = {}
    for p in products:
        if p.category not in categories:
            categories[p.category] = []
        categories[p.category].append(p.width)
    
    print(f"  Total products: {len(products)}")
    print(f"  Width range: {min(widths)} - {max(widths)} cm")
    print(f"  GCD of all widths: {find_gcd(widths)}")
    print(f"  Categories: {list(categories.keys())}")
    
    # Show width patterns
    print("\n  Width patterns by category:")
    for cat, cat_widths in categories.items():
        unique_widths = sorted(set(cat_widths))
        print(f"    {cat}: {unique_widths}")

def run_experiment(n_products, capacity):
    """Run a single experiment with all algorithms"""
    print(f"\nRunning experiment: {n_products} products, {capacity}cm shelf")
    print("-" * 60)
    
    # Generate products
    products = generate_products(n_products)
    
    # Analyze products
    analyze_products(products)
    
    # Run all algorithms
    results = []
    
    # 1. Optimized Dynamic Programming
    print("\nRunning Optimized DP...")
    dp_result = solve_dp(products, capacity)
    results.append(dp_result)
    print(f"  Done! Profit: ${dp_result.total_profit:.2f}, Time: {dp_result.computation_time*1000:.2f}ms")
    
    
    # 5. Hybrid LLM-DP
    print("Running Hybrid LLM-DP...")
    hybrid_result = solve_hybrid_llm_dp(products, capacity)
    results.append(hybrid_result)
    print(f"  Done! Profit: ${hybrid_result.total_profit:.2f}, Time: {hybrid_result.computation_time*1000:.2f}ms")
    
    # Performance comparison
    print("\nPerformance vs Optimal (DP):")
    for result in results:
        if result.method != "Optimized DP":
            performance = (result.total_profit / dp_result.total_profit) * 100
            print(f"  {result.method}: {performance:.1f}% of optimal")
    
    return results, products, capacity

def main():
    """Run all experiments"""
    print("="*60)
    print("RETAIL SHELF SPACE ALLOCATION EXPERIMENTS")
    print("="*60)
    
    # Create output directory
    os.makedirs("results", exist_ok=True)
    
    # Run experiments with different sizes
    experiments = [
        (20, 200),   # Small: 20 products, 200cm shelf
        (50, 400),   # Medium: 50 products, 400cm shelf
        (100, 800),  # Large: 100 products, 800cm shelf
    ]
    
    all_results = []
    
    for n_products, capacity in experiments:
        results, products, cap = run_experiment(n_products, capacity)
        all_results.append((results, products, cap))
    
    # Use last experiment for detailed visualization
    last_results, last_products, last_capacity = all_results[-1]
    
    for curr_result,curr_prod,curr_capacity in all_results:
        print_summary(curr_result,curr_capacity)
    
    # 1. All methods comparison
        plot_profit_comparison(curr_result, "results/"+str(len(curr_prod))+"_profit_comparison_all.png")
    
    # 2. DP vs LLM focused comparison
        plot_dp_vs_llm_comparison(curr_result, "results/"+str(len(curr_prod))+"_dp_vs_llm_comparison.png")
    
    # 3. Runtime comparison
        plot_runtime_comparison(curr_result, "results/"+str(len(curr_prod))+"_runtime_comparison.png")
    
    # 4. Utilization comparison
        plot_utilization_comparison(curr_result, curr_capacity, "results/"+str(len(curr_prod))+"_utilization_comparison.png")
        save_results_csv(curr_result, curr_prod, curr_capacity, "results/"+str(len(curr_prod))+"_results_summary.csv")
    
    # 5. Scalability analysis (if multiple experiments)
    if len(all_results) > 1:
        plot_scalability_analysis(all_results, "results/scalability_analysis.png")
    
    # Save results to CSV
        
    
    print("\nAll experiments completed!")
    print("Results saved in 'results/' directory")
    
    # Analysis insights
    print("\n" + "="*60)
    print("INSIGHTS")
    print("="*60)
    print("1. When products have good GCD (e.g., all multiples of 8), DP is very fast")
    print("2. When products have mixed widths (GCD=1), DP becomes slower")
    print("3. Hybrid approach groups products to improve GCD within groups")
    print("5. Parallel processing in hybrid can further improve speed")

if __name__ == "__main__":
    main()# main.py
"""Main script to run all experiments"""

import os
from data_generator import generate_products
from optimized_dp import solve_dp
from pure_llm import solve_pure_llm
from hybrid_llm_dp import solve_hybrid_llm_dp
from visualizer import (plot_profit_comparison, plot_runtime_comparison, 
                       plot_utilization_comparison, save_results_csv, print_summary)

def run_experiment(n_products, capacity):
    """Run a single experiment with all algorithms"""
    print(f"\nRunning experiment: {n_products} products, {capacity}cm shelf")
    print("-" * 60)
    
    # Generate products
    products = generate_products(n_products)
    
    # Run all algorithms
    results = []
    
    # 1. Optimized Dynamic Programming
    print("Running Optimized DP...")
    dp_result = solve_dp(products, capacity)
    results.append(dp_result)
    print(f"  Done! Profit: ${dp_result.total_profit:.2f}")
    
    
    # 5. Hybrid LLM-DP
    print("Running Hybrid LLM-DP...")
    hybrid_result = solve_hybrid_llm_dp(products, capacity)
    results.append(hybrid_result)
    print(f"  Done! Profit: ${hybrid_result.total_profit:.2f}")
    
    return results, products, capacity

def main():
    """Run all experiments"""
    print("="*60)
    print("RETAIL SHELF SPACE ALLOCATION EXPERIMENTS")
    print("="*60)
    
    # Create output directory
    os.makedirs("results", exist_ok=True)
    
    # Run experiments with different sizes
    experiments = [
        (20, 200),   # Small: 20 products, 200cm shelf
        (50, 400),   # Medium: 50 products, 400cm shelf
        (100, 800),  # Large: 100 products, 800cm shelf
    ]
    
    all_results = []
    
    for n_products, capacity in experiments:
        results, products, cap = run_experiment(n_products, capacity)
        all_results.append((results, products, cap))
    
    
    print("\nAll experiments completed!")
    print("Results saved in 'results/' directory")

if __name__ == "__main__":
    main()