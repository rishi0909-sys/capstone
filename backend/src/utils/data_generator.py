import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

def generate_synthetic_data(num_transactions=5000, output_path="data/sample_transactions.csv"):
    print("Generating SYNTHETIC DEMO DATA...")
    np.random.seed(42)
    random.seed(42)
    
    # Products and Categories
    products = {
        "Electronics": [
            "Laptop - Standard", "Laptop - Pro", "Wireless Mouse", "USB-C Hub", "Monitor 24in", "Monitor 27in"
        ],
        "Gaming Accessories": [
            "Gaming Laptop", "Gaming Mouse", "Mechanical Keyboard", "Gaming Headset", "RGB Mousepad"
        ],
        "Office Supplies": [
            "Notebook", "Pens (12 Pack)", "Desk Lamp", "Office Chair", "Whiteboard"
        ]
    }
    
    flat_products = []
    product_categories = {}
    product_prices = {}
    
    for cat, items in products.items():
        for item in items:
            flat_products.append(item)
            product_categories[item] = cat
            # Assign random base prices
            if "Laptop" in item:
                product_prices[item] = round(random.uniform(600, 2500), 2)
            elif "Monitor" in item:
                product_prices[item] = round(random.uniform(150, 400), 2)
            elif "Keyboard" in item:
                product_prices[item] = round(random.uniform(50, 150), 2)
            else:
                product_prices[item] = round(random.uniform(10, 80), 2)

    # Date range (last 1 year)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    transactions = []
    
    customer_ids = [f"CUST_{i:04d}" for i in range(1, 1001)]
    # Demographics for customers
    customer_demographics = {}
    for cid in customer_ids:
        age = int(np.random.normal(35, 12))
        age = max(18, min(age, 80))
        customer_demographics[cid] = {
            "age": age,
            "gender": random.choice(["Male", "Female", "Other"]),
            "region": random.choice(["North", "South", "East", "West"])
        }

    transaction_id_counter = 1
    
    # Pre-calculate seasonal probabilities (e.g., Mechanical Keyboard spike in Dec)
    def is_holiday_season(date):
        return date.month == 11 and date.day > 15 or date.month == 12

    for _ in range(num_transactions):
        # Transaction context
        t_date = start_date + timedelta(days=random.randint(0, 365), hours=random.randint(8, 22))
        cid = random.choice(customer_ids)
        age = customer_demographics[cid]["age"]
        
        # Determine basket size
        basket_size = max(1, int(np.random.normal(2.5, 1.5)))
        basket = []
        
        # INJECTED PATTERNS:
        # 1. Laptop -> Wireless Mouse
        # 2. Gaming Laptop -> Gaming Mouse / Gaming Headset
        # 3. Mechanical Keyboard seasonal spike (Dec)
        # 4. Gaming related items bought more by younger demographics (18-25)
        
        base_item = random.choice(flat_products)
        
        # Override base item based on seasonality/demographics to inject patterns
        if is_holiday_season(t_date) and random.random() < 0.3:
            base_item = "Mechanical Keyboard"
            
        if age <= 25 and random.random() < 0.4:
            base_item = "Gaming Laptop"
            
        basket.append(base_item)
        
        # Add correlated items
        if "Laptop - Standard" in basket or "Laptop - Pro" in basket:
            if random.random() < 0.7:  # 70% chance to buy mouse with normal laptop
                basket.append("Wireless Mouse")
        
        if "Gaming Laptop" in basket:
            if random.random() < 0.6:
                basket.append("Gaming Mouse")
            if random.random() < 0.5:
                basket.append("Gaming Headset")
                
        # Fill rest of basket randomly
        while len(basket) < basket_size and len(basket) < len(flat_products):
            random_item = random.choice(flat_products)
            if random_item not in basket:
                basket.append(random_item)
                
        # Remove duplicates
        basket = list(set(basket))
        
        tid = f"TRX_{transaction_id_counter:06d}"
        
        for item in basket:
            transactions.append({
                "transaction_id": tid,
                "customer_id": cid,
                "date": t_date.strftime("%Y-%m-%d %H:%M:%S"),
                "product_name": item,
                "category": product_categories[item],
                "price": product_prices[item],
                "quantity": random.randint(1, 3) if "Notebook" in item or "Pens" in item else 1,
                "age": customer_demographics[cid]["age"],
                "gender": customer_demographics[cid]["gender"],
                "region": customer_demographics[cid]["region"]
            })
            
        transaction_id_counter += 1

    df = pd.DataFrame(transactions)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} rows across {num_transactions} transactions.")
    print(f"Saved to {output_path}")
    print("WARNING: This is SYNTHETIC DEMO DATA intended for demonstration purposes only.")

if __name__ == "__main__":
    generate_synthetic_data()
