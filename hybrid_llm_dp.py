# hybrid_llm_dp.py
"""Hybrid approach combining LLM categorization with DP"""

import time
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from product import Result
from optimized_dp import solve_dp

# Check for OpenAI
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

def categorize_products_fallback(products):
    """Fallback grouping based on width patterns"""
    groups = {}
    
    # Group by common factors
    groups['width_8_group'] = [p for p in products if p.width % 8 == 0]
    groups['width_6_group'] = [p for p in products if p.width % 6 == 0 and p.width % 8 != 0]
    groups['width_5_group'] = [p for p in products if p.width % 5 == 0 and p.width % 6 != 0 and p.width % 8 != 0]
    
    # Collect remaining products
    used_ids = set()
    for group in groups.values():
        used_ids.update(p.id for p in group)
    
    groups['other_widths'] = [p for p in products if p.id not in used_ids]
    
    # Remove empty groups
    return {k: v for k, v in groups.items() if v}

def categorize_with_llm(products):
    """Use LLM to group products intelligently"""
    if not OPENAI_AVAILABLE or not os.getenv("OPENAI_API_KEY"):
        return categorize_products_fallback(products)
    
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        
        # Create product list for LLM
        product_list = []
        for p in products[:40]:  # Limit for prompt
            product_list.append(f"ID:{p.id}, Width:{p.width}cm")
        
        prompt = f"""Group these products for optimal mathematical optimization.

PRODUCTS:
{chr(10).join(product_list)}

GROUPING RULES:
1. Products with widths that share common factors should be grouped together
2. For example: [8,16,24] share factor 8, [10,15,20] share factor 5
3. Products with prime widths should be in their own group

Create groups that maximize the GCD (greatest common divisor) within each group.

Return as JSON: {{"group_name": [product_ids]}}
Example: {{"multiples_of_8": [0,3,7], "multiples_of_5": [1,4,9]}}"""
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert at finding mathematical patterns."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=800
        )
        
        # Parse response
        response_text = response.choices[0].message.content
        start = response_text.find('{')
        end = response_text.rfind('}') + 1
        
        if start == -1 or end == 0:
            return categorize_products_fallback(products)
        
        groups_data = json.loads(response_text[start:end])
        
        # Convert to product groups
        product_dict = {p.id: p for p in products}
        groups = {}
        used_ids = set()
        
        for group_name, product_ids in groups_data.items():
            group_products = []
            for pid in product_ids:
                if int(pid) in product_dict and int(pid) not in used_ids:
                    group_products.append(product_dict[int(pid)])
                    used_ids.add(int(pid))
            if group_products:
                groups[group_name] = group_products
        
        # Add ungrouped products
        ungrouped = [p for p in products if p.id not in used_ids]
        if ungrouped:
            groups['ungrouped'] = ungrouped
        
        return groups
        
    except:
        return categorize_products_fallback(products)

def solve_group_dp(group_name, group_products, group_capacity):
    """Solve DP for a single group (for parallel execution)"""
    result = solve_dp(group_products, group_capacity)
    return group_name, result

def solve_hybrid_llm_dp(products, capacity, use_parallel=True):
    """Solve using hybrid LLM+DP approach with optional parallel processing"""
    
    
    # Step 1: Categorize products
    groups = categorize_with_llm(products)
    print(f"Hybrid: Created {len(groups)} groups")
    
    # Step 2: Calculate capacity allocation
    group_info = {}
    total_density = 0
    
    for group_name, group_products in groups.items():
        avg_density = sum(p.profit / p.width for p in group_products) / len(group_products)
        group_info[group_name] = {
            'products': group_products,
            'density': avg_density,
            'count': len(group_products)
        }
        total_density += avg_density
    
    # Allocate capacity based on density
    for group_name in group_info:
        group_info[group_name]['capacity'] = int(
            capacity * group_info[group_name]['density'] / total_density
        )
    
    # Step 3: Solve DP for each group
    all_allocations = {}
    total_profit = 0
    total_width = 0

    start_time = time.time()
    
    if use_parallel and len(groups) > 1:
        # Parallel execution
        with ThreadPoolExecutor(max_workers=min(len(groups), 4)) as executor:
            futures = []
            for group_name, info in group_info.items():
                future = executor.submit(
                    solve_group_dp, 
                    group_name, 
                    info['products'], 
                    info['capacity']
                )
                futures.append(future)
            
            # Collect results
            for future in as_completed(futures):
                group_name, result = future.result()
                all_allocations.update(result.allocations)
                total_profit += result.total_profit
                total_width += result.total_width
    else:
        # Sequential execution
        for group_name, info in group_info.items():
            result = solve_dp(info['products'], info['capacity'])
            all_allocations.update(result.allocations)
            total_profit += result.total_profit
            total_width += result.total_width
    
    computation_time = time.time() - start_time
    
    return Result("Hybrid LLM-DP", all_allocations, total_profit, total_width, computation_time)