# pure_llm.py
"""Pure LLM approach for shelf allocation"""

import time
import json
import os
from product import Result

# Simple check for OpenAI
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("OpenAI not installed. Pure LLM will return empty results.")

def create_llm_prompt(products, capacity):
    """Create an optimized prompt for the LLM"""
    # Sort products by profit density to help LLM
    product_info = []
    for p in products[:30]:  # Limit for prompt size
        density = p.profit / p.width
        product_info.append((density, p))
    
    # Sort by density descending
    product_info.sort(reverse=True)
    
    # Create detailed product descriptions
    product_desc = []
    for density, p in product_info:
        product_desc.append(
            f"- Product {p.id}: width={p.width}cm, profit=${p.profit}/facing, "
            f"max={p.max_facings}, profit_per_cm=${density:.3f}"
        )
    
    prompt = f"""You are solving a shelf space allocation problem. 

PROBLEM:
- Shelf capacity: {capacity}cm
- Goal: Maximize total profit
- Constraint: Total width must not exceed {capacity}cm

PRODUCTS (sorted by profit density):
{chr(10).join(product_desc)}

Choose the products and respective facings based on the input data, give me back the combinations of json in a way i get maximum profit when we calculate.

IMPORTANT: Calculate total width as you go. The sum of (facings * width) for all products must be less than or equal to {capacity}cm.
I need maximum profit out of the combinations. 
Return ONLY a JSON object with product IDs and facings.
Example: {{"0": 3, "5": 2, "10": 1}}"""
    
    return prompt

def parse_llm_response(response_text, products):
    """Extract allocations from LLM response"""
    try:
        # Find JSON in response
        start = response_text.find('{')
        end = response_text.rfind('}') + 1
        if start == -1 or end == 0:
            return {}
        
        json_str = response_text[start:end]
        allocations_raw = json.loads(json_str)
        
        # Convert to proper format with validation
        allocations = {}
        product_dict = {p.id: p for p in products}
        
        for product_id_str, facings in allocations_raw.items():
            try:
                product_id = int(product_id_str)
                facings = int(facings)
                
                # Validate
                if product_id in product_dict and 0 < facings <= product_dict[product_id].max_facings:
                    allocations[product_id] = facings
            except:
                continue
        
        return allocations
    except Exception as e:
        print(f"Parse error: {e}")
        return {}

def solve_pure_llm(products, capacity):
    """Solve using pure LLM approach - no fallback"""
    start_time = time.time()
    
    # Check if OpenAI is available
    api_key = os.getenv("OPENAI_API_KEY")
    if not OPENAI_AVAILABLE or not api_key:
        print("Pure LLM: No OpenAI API available")
        computation_time = time.time() - start_time
        return Result("Pure LLM", {}, 0, 0, computation_time)
    
    try:
        openai.api_key = api_key
        
        # Create optimized prompt
        prompt = create_llm_prompt(products, capacity)
        
        # Call LLM with specific parameters for better results
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert at optimization problems. Always verify constraints."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,  # More deterministic
            max_tokens=1000   # More space for larger problems
        )
        
        # Parse response
        response_text = response.choices[0].message.content
        allocations = parse_llm_response(response_text, products)
        
        # Calculate metrics
        total_profit = sum(allocations.get(p.id, 0) * p.profit for p in products)
        total_width = sum(allocations.get(p.id, 0) * p.width for p in products)
        
        # Log result
        if total_width > capacity:
            print(f"Pure LLM: Solution exceeds capacity ({total_width} > {capacity})")
        else:
            print(f"Pure LLM: Valid solution with {len(allocations)} products")
        
        computation_time = time.time() - start_time
        return Result("Pure LLM", allocations, total_profit, total_width, computation_time)
        
    except Exception as e:
        print(f"Pure LLM error: {e}")
        computation_time = time.time() - start_time
        return Result("Pure LLM", {}, 0, 0, computation_time)