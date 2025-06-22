# data_generator.py
"""Generate synthetic product data with more realistic patterns"""

import random
from product import Product

def generate_products(n_products=50, seed=42):
    """Generate synthetic product data with patterns that benefit hybrid approach"""
    random.seed(seed)
    
    products = []
    product_id = 0
    
    # Create product groups with shared characteristics
    # This mimics real retail where products in same category often have similar dimensions
    
    # Group 1: Standard cans (all 8cm width - great for DP optimization)
    num_cans = n_products // 5
    base_profit_can = 3.0
    for i in range(num_cans):
        products.append(Product(
            id=product_id,
            name=f"Can_Product_{i}",
            category="Canned Goods",
            width=8,  # All same width - perfect for GCD optimization
            profit=round(base_profit_can + random.uniform(-0.5, 2.0), 2),
            max_facings=random.randint(3, 6)
        ))
        product_id += 1
    
    # # Group 2: Bottles (widths are multiples of 5)
    num_bottles = n_products // 5
    bottle_widths = [10, 15, 20]  # All divisible by 5
    base_profit_bottle = 5.0
    for i in range(num_bottles):
        products.append(Product(
            id=product_id,
            name=f"Bottle_Product_{i}",
            category="Beverages",
            width=random.choice(bottle_widths),
            profit=round(base_profit_bottle + random.uniform(-1.0, 3.0), 2),
            max_facings=random.randint(2, 5)
        ))
        product_id += 1
    
    # Group 3: Premium items (high profit, odd widths)
    num_premium = n_products // 5
    premium_widths = [7, 11, 13, 17]  # Prime or odd numbers - bad for GCD
    base_profit_premium = 8.0
    for i in range(num_premium):
        products.append(Product(
            id=product_id,
            name=f"Premium_Product_{i}",
            category="Premium",
            width=random.choice(premium_widths),
            profit=round(base_profit_premium + random.uniform(0, 4.0), 2),
            max_facings=random.randint(2, 4)
        ))
        product_id += 1
    
    # Group 4: Snacks (multiples of 6)
    num_snacks = n_products // 5
    snack_widths = [6, 12, 18]  # All divisible by 6
    base_profit_snack = 4.0
    for i in range(num_snacks):
        products.append(Product(
            id=product_id,
            name=f"Snack_Product_{i}",
            category="Snacks",
            width=random.choice(snack_widths),
            profit=round(base_profit_snack + random.uniform(-0.5, 1.5), 2),
            max_facings=random.randint(3, 7)
        ))
        product_id += 1
    
    # # Group 5: Mixed items (random widths)
    remaining = n_products - product_id
    for i in range(remaining):
        category = random.choice(["Dairy", "Bakery", "Other"])
        products.append(Product(
            id=product_id,
            name=f"{category}_Product_{i}",
            category=category,
            width=random.randint(5, 25),
            profit=round(random.uniform(1.0, 9.0), 2),
            max_facings=random.randint(2, 8)
        ))
        product_id += 1
    
    # Shuffle to mix products
    random.shuffle(products)
    
    # Reassign IDs after shuffle
    for i, product in enumerate(products):
        product.id = i
    
    return products