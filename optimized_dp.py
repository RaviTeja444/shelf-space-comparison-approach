

import time
import math
from product import Result

def find_gcd(numbers):
    """Find GCD of a list of numbers"""
    result = numbers[0]
    for i in range(1, len(numbers)):
        result = math.gcd(result, numbers[i])
    return result

def solve_dp(products, capacity):
    """Solve shelf allocation using optimized dynamic programming"""
    start_time = time.time()
    
    n = len(products)
    if n == 0:
        return Result("Optimized DP", {}, 0, 0, 0)
    
    # Find GCD of all product widths
    widths = [p.width for p in products]
    gcd = find_gcd(widths)
    # Reduce the problem size using GCD
    reduced_capacity = capacity // gcd
    reduced_widths = [w // gcd for w in widths]
    
    # Create DP table
    # dp[i][j] = maximum profit using first i products with capacity j
    dp = [[0 for _ in range(reduced_capacity + 1)] for _ in range(n + 1)]
    
    # Keep track of how many facings of each product
    parent = [[0 for _ in range(reduced_capacity + 1)] for _ in range(n + 1)]
    
    # Fill DP table
    for i in range(1, n + 1):
        product = products[i - 1]
        w = reduced_widths[i - 1]
        p = product.profit
        
        for c in range(reduced_capacity + 1):
            # Option 1: Don't take this product
            dp[i][c] = dp[i - 1][c]
            parent[i][c] = 0
            
            # Option 2: Take some facings of this product
            for facings in range(1, min(product.max_facings + 1, c // w + 1)):
                if facings * w <= c:
                    value = dp[i - 1][c - facings * w] + facings * p
                    if value > dp[i][c]:
                        dp[i][c] = value
                        parent[i][c] = facings
    
    # Backtrack to find which products to include
    allocations = {}
    c = reduced_capacity
    
    for i in range(n, 0, -1):
        facings = parent[i][c]
        if facings > 0:
            product = products[i - 1]
            allocations[product.id] = facings
            c -= facings * reduced_widths[i - 1]
    
    # Calculate total profit and width
    total_profit = dp[n][reduced_capacity]
    total_width = sum(allocations.get(p.id, 0) * p.width for p in products)
    
    computation_time = time.time() - start_time
    
    return Result("Optimized DP", allocations, total_profit, total_width, computation_time)