#!/usr/bin/env python3
"""
Online Retail II Dataset Converter & Enricher
---------------------------------------------
Converts 'online_retail_II.xlsx' into a clean, normalized CSV enriched with:
- category (inferred via product description keywords)
- age (consistently generated per customer, 18-75)
- gender (consistently generated per customer)
- region (UK regions North/South/East/West + International)

Usage:
    python3 convert_and_enrich.py [--mode sample|year2|all] [--max-rows N]
"""

import sys
import os
import argparse
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import random
import hashlib
import csv

DEFAULT_INPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "online_retail_II.xlsx")
DEFAULT_OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend", "data", "online_retail_clean.csv")

# Non-product stock codes to filter out
ADMIN_CODES = {
    "POST", "D", "M", "PADS", "DOT", "CRUK", "BANK CHARGES", 
    "AMAZON FEE", "SAMP", "ADJUST", "ADJUST2", "TEST"
}

CATEGORY_KEYWORDS = {
    "Seasonal & Holiday": [
        "christmas", "xmas", "tree", "easter", "halloween", "decoration", 
        "santa", "snow", "advent", "holly", "reindeer", "noel"
    ],
    "Home & Living": [
        "heart", "cushion", "clock", "frame", "mirror", "candle", "t-light", 
        "holder", "doormat", "wall", "sign", "vase", "plaque", "light", 
        "lights", "lamp", "curtain", "lantern", "rug", "hook"
    ],
    "Kitchen & Dining": [
        "mug", "plate", "bowl", "cup", "saucer", "teapot", "cutlery", 
        "glass", "bottle", "jar", "baking", "kitchen", "tray", "tin", 
        "coffee", "tea", "cake", "spoon", "fork", "dish", "jug", "pan"
    ],
    "Stationery & Crafts": [
        "pen", "pencil", "notebook", "card", "wrap", "sticker", "tape", 
        "paper", "calendar", "box", "envelope", "stamp", "craft", "chalkboard",
        "rubber", "eraser", "ruler", "scissors"
    ],
    "Fashion & Accessories": [
        "bag", "tote", "purse", "wallet", "scarf", "glove", "sock", 
        "apron", "jewellery", "necklace", "bracelet", "charm", "earring",
        "pendant", "umbrella"
    ],
    "Toys & Games": [
        "toy", "game", "puzzle", "doll", "plush", "teddy", "child", 
        "kid", "mini", "ball", "play", "circus", "magic"
    ]
}

def categorize_product(desc: str) -> str:
    if not desc:
        return "General Retail"
    desc_lower = desc.lower()
    for cat, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in desc_lower for kw in keywords):
            return cat
    return "General Retail"

def get_customer_demographics(cust_id: str):
    """Deterministically assign age, gender, and UK region per customer."""
    # Use MD5 hash of customer_id for reproducible pseudo-random values
    h = int(hashlib.md5(cust_id.encode('utf-8')).hexdigest()[:8], 16)
    
    # Age: Normal distribution around 37 (standard UK online shoppers)
    # Map hash to 0..1
    u1 = (h % 10000) / 10000.0
    u2 = ((h // 10000) % 10000) / 10000.0
    u1 = max(0.0001, min(0.9999, u1))
    
    # Box-Muller transform for normal distribution
    import math
    z0 = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
    age = int(37 + z0 * 12)
    age = max(18, min(age, 75))
    
    # Gender
    g_val = (h >> 4) % 100
    if g_val < 54:
        gender = "Female"
    elif g_val < 92:
        gender = "Male"
    else:
        gender = "Other"
        
    # Region for UK
    reg_val = (h >> 8) % 4
    regions = ["North", "South", "East", "West"]
    region = regions[reg_val]
    
    return age, gender, region

def excel_serial_to_datetime_str(val_str: str) -> str:
    try:
        days = float(val_str)
        # Excel 1900 date system epoch
        base = datetime(1899, 12, 30)
        dt = base + timedelta(days=days)
        return dt.strftime('%Y-%m-%d %H:%M:%S')
    except Exception:
        return ""

def load_shared_strings(z: zipfile.ZipFile):
    print("  [1/3] Indexing shared strings table...")
    shared_strings = []
    if 'xl/sharedStrings.xml' in z.namelist():
        with z.open('xl/sharedStrings.xml') as f:
            for event, elem in ET.iterparse(f, events=('end',)):
                if elem.tag.endswith('si'):
                    t = elem.find('{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t')
                    if t is not None and t.text:
                        shared_strings.append(t.text)
                    else:
                        parts = [p.text for p in elem.findall('.//{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t') if p.text]
                        shared_strings.append("".join(parts))
                    elem.clear()
    print(f"  --> Loaded {len(shared_strings):,} unique strings.")
    return shared_strings

def process_sheet(z: zipfile.ZipFile, sheet_xml: str, shared_strings: list, writer: csv.writer, 
                  max_rows: int = None, current_count: int = 0):
    print(f"  [2/3] Processing rows from {sheet_xml}...")
    
    kept = 0
    skipped_cancel = 0
    skipped_bad_qty_price = 0
    skipped_admin = 0
    skipped_missing = 0
    
    row_idx = 0
    with z.open(sheet_xml) as f:
        for event, elem in ET.iterparse(f, events=('end',)):
            if elem.tag.endswith('row'):
                row_idx += 1
                if row_idx == 1:
                    elem.clear()
                    continue # header
                    
                row_dict = {}
                for c in elem.findall('{http://schemas.openxmlformats.org/spreadsheetml/2006/main}c'):
                    ref = c.attrib.get('r', '')
                    col = "".join([ch for ch in ref if ch.isalpha()])
                    val_elem = c.find('{http://schemas.openxmlformats.org/spreadsheetml/2006/main}v')
                    cell_type = c.attrib.get('t')
                    val = val_elem.text if val_elem is not None else ''
                    if cell_type == 's' and val.isdigit():
                        val = shared_strings[int(val)]
                    row_dict[col] = val
                    
                elem.clear()
                
                # Column mapping:
                # A: Invoice, B: StockCode, C: Description, D: Quantity, E: InvoiceDate, F: Price, G: Customer ID, H: Country
                invoice = row_dict.get('A', '').strip()
                stock_code = row_dict.get('B', '').strip().upper()
                desc = row_dict.get('C', '').strip()
                qty_str = row_dict.get('D', '0')
                date_str = row_dict.get('E', '')
                price_str = row_dict.get('F', '0')
                cust_id_raw = row_dict.get('G', '').strip()
                country = row_dict.get('H', '').strip()
                
                # Filter 1: Invoices
                if not invoice or invoice.startswith(('C', 'c', 'A', 'a')):
                    skipped_cancel += 1
                    continue
                    
                # Filter 2: Non-product administrative codes
                if stock_code in ADMIN_CODES or any(stock_code.startswith(ac) for ac in ["POST", "BANK", "AMAZON"]):
                    skipped_admin += 1
                    continue
                    
                # Filter 3: Missing description
                if not desc or len(desc) < 2:
                    skipped_missing += 1
                    continue
                    
                # Filter 4: Positive Quantity & Price
                try:
                    qty = int(float(qty_str))
                    price = round(float(price_str), 2)
                except ValueError:
                    skipped_bad_qty_price += 1
                    continue
                    
                if qty <= 0 or price <= 0.0:
                    skipped_bad_qty_price += 1
                    continue
                    
                # Date format
                dt_formatted = excel_serial_to_datetime_str(date_str)
                if not dt_formatted:
                    continue
                    
                # Customer ID & Demographics
                if cust_id_raw:
                    # Clean customer id (sometimes has decimal like 13085.0)
                    cust_clean = cust_id_raw.split('.')[0]
                    customer_id = f"CUST_{int(cust_clean):05d}"
                else:
                    # Deterministic guest ID based on invoice
                    customer_id = f"GUEST_{invoice}"
                    
                age, gender, uk_region = get_customer_demographics(customer_id)
                
                # Region: keep UK regions or use Country
                if country in ["United Kingdom", "UK", ""]:
                    region = uk_region
                else:
                    region = country
                    
                # Category
                category = categorize_product(desc)
                
                # Clean product name (capitalize nicely)
                product_name = desc.title()
                
                # Write row
                # Schema: transaction_id,customer_id,date,product_name,category,price,quantity,age,gender,region
                writer.writerow([
                    invoice,
                    customer_id,
                    dt_formatted,
                    product_name,
                    category,
                    price,
                    qty,
                    age,
                    gender,
                    region
                ])
                
                kept += 1
                if max_rows and (current_count + kept) >= max_rows:
                    break
                    
    print(f"  --> Kept: {kept:,} rows.")
    print(f"  --> Filtered out: {skipped_cancel:,} cancelled, {skipped_bad_qty_price:,} zero/neg qty/price, {skipped_admin:,} admin, {skipped_missing:,} missing desc.")
    return kept

def main():
    parser = argparse.ArgumentParser(description="Clean and enrich Online Retail II dataset.")
    parser.add_argument("--input", "-i", default=DEFAULT_INPUT, help="Path to online_retail_II.xlsx")
    parser.add_argument("--output", "-o", default=DEFAULT_OUTPUT, help="Output clean CSV path")
    parser.add_argument("--mode", "-m", choices=["sample", "year2", "all"], default="year2",
                        help="'sample' (50k rows), 'year2' (2010-2011, ~400k rows, recommended), 'all' (both years, ~800k rows)")
    parser.add_argument("--max-rows", "-n", type=int, default=None, help="Explicit max rows cap")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        sys.exit(1)
        
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    max_rows = args.max_rows
    if args.mode == "sample" and not max_rows:
        max_rows = 50000
        
    sheets_to_process = []
    if args.mode == "all":
        sheets_to_process = [
            ("Year 2009-2010", "xl/worksheets/sheet1.xml"),
            ("Year 2010-2011", "xl/worksheets/sheet2.xml")
        ]
    elif args.mode in ["year2", "sample"]:
        # Year 2010-2011 is sheet2, the most referenced period in academic ML benchmarks
        sheets_to_process = [
            ("Year 2010-2011", "xl/worksheets/sheet2.xml")
        ]
        
    print(f"Starting conversion of: {args.input}")
    print(f"Mode: {args.mode} | Output: {args.output}")
    if max_rows:
        print(f"Row limit: {max_rows:,}")
        
    with zipfile.ZipFile(args.input, 'r') as z:
        shared_strings = load_shared_strings(z)
        
        with open(args.output, 'w', newline='', encoding='utf-8') as out_f:
            writer = csv.writer(out_f)
            # Write header
            writer.writerow([
                "transaction_id", "customer_id", "date", "product_name", 
                "category", "price", "quantity", "age", "gender", "region"
            ])
            
            total_written = 0
            for name, sheet_xml in sheets_to_process:
                print(f"\nProcessing {name} ({sheet_xml})...")
                written = process_sheet(
                    z, sheet_xml, shared_strings, writer, 
                    max_rows=max_rows, current_count=total_written
                )
                total_written += written
                if max_rows and total_written >= max_rows:
                    break
                    
    file_size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"\n[DONE] Successfully wrote {total_written:,} clean & enriched rows to:")
    print(f"  {args.output} ({file_size_mb:.2f} MB)")
    print("\nNext steps:")
    print("  1. You can upload this CSV via the web interface or replace backend/data/sample_transactions.csv.")
    print("  2. Run the preprocessing, FP-Growth, forecasting, and demographic pipelines!")

if __name__ == "__main__":
    main()
