#!/usr/bin/env python3
"""
Dataset 3 Cleaner & Normalizer
==============================
Source: docs/dataset3.zip -> /tmp/dataset3_extracted/online_retail_II.xlsx
Processes both Year 2009-2010 and Year 2010-2011 (full 2-year authentic benchmark).
Normalizes to the unified engine schema:
['transaction_id', 'customer_id', 'date', 'product_name', 'category', 'price', 'quantity', 'revenue', 'country']
Handles guest checkouts synthetically ('GUEST_<transaction_id>') to preserve transaction volume.
"""

import os
import sys
import zipfile
import csv
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

RAW_XLSX_PATH = "/tmp/dataset3_extracted/online_retail_II.xlsx"
OUTPUT_CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "dataset_3_clean.csv")

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
        return "General Merchandise"
    desc_lower = desc.lower()
    for cat, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in desc_lower for kw in keywords):
            return cat
    return "General Merchandise"

def excel_serial_to_datetime_str(val_str: str) -> str:
    try:
        days = float(val_str)
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

def main():
    print("=" * 80)
    print("  DATASET 3 PREPARATION (UCI Online Retail II - Full 2-Year Dataset)")
    print("=" * 80)

    if not os.path.exists(RAW_XLSX_PATH):
        print(f"Error: Raw file not found at {RAW_XLSX_PATH}")
        sys.exit(1)

    os.makedirs(os.path.dirname(OUTPUT_CSV_PATH), exist_ok=True)

    sheets = [
        ("Year 2009-2010", "xl/worksheets/sheet1.xml"),
        ("Year 2010-2011", "xl/worksheets/sheet2.xml")
    ]

    total_raw_rows = 0
    total_cancelled = 0
    total_bad_price_qty = 0
    total_admin = 0
    total_missing_desc = 0
    total_missing_cust = 0
    total_clean_rows = 0

    with zipfile.ZipFile(RAW_XLSX_PATH, 'r') as z:
        shared_strings = load_shared_strings(z)

        with open(OUTPUT_CSV_PATH, 'w', newline='', encoding='utf-8') as out_f:
            writer = csv.writer(out_f)
            writer.writerow([
                "transaction_id", "customer_id", "date", "product_name",
                "category", "price", "quantity", "revenue", "country"
            ])

            for sheet_name, sheet_xml in sheets:
                print(f"\nProcessing {sheet_name} ({sheet_xml})...")
                row_idx = 0
                kept_in_sheet = 0

                with z.open(sheet_xml) as f:
                    for event, elem in ET.iterparse(f, events=('end',)):
                        if elem.tag.endswith('row'):
                            row_idx += 1
                            if row_idx == 1:
                                elem.clear()
                                continue  # Skip header

                            total_raw_rows += 1
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

                            # Columns:
                            # A: Invoice, B: StockCode, C: Description, D: Quantity, E: InvoiceDate, F: Price, G: Customer ID, H: Country
                            invoice = row_dict.get('A', '').strip()
                            stock_code = row_dict.get('B', '').strip().upper()
                            desc = row_dict.get('C', '').strip()
                            qty_str = row_dict.get('D', '0')
                            date_str = row_dict.get('E', '')
                            price_str = row_dict.get('F', '0')
                            cust_id_raw = row_dict.get('G', '').strip()
                            country = row_dict.get('H', '').strip() or "United Kingdom"

                            # Cancellations
                            if not invoice or invoice.startswith(('C', 'c', 'A', 'a')):
                                total_cancelled += 1
                                continue

                            # Administrative codes
                            if stock_code in ADMIN_CODES or any(stock_code.startswith(ac) for ac in ["POST", "BANK", "AMAZON"]):
                                total_admin += 1
                                continue

                            # Missing description
                            if not desc or len(desc) < 2:
                                total_missing_desc += 1
                                continue

                            # Positive price & quantity
                            try:
                                qty = int(float(qty_str))
                                price = round(float(price_str), 2)
                            except ValueError:
                                total_bad_price_qty += 1
                                continue

                            if qty <= 0 or price <= 0.0:
                                total_bad_price_qty += 1
                                continue

                            # Date parse
                            dt_formatted = excel_serial_to_datetime_str(date_str)
                            if not dt_formatted:
                                continue

                            # Customer ID mapping
                            if cust_id_raw:
                                cust_clean = cust_id_raw.split('.')[0]
                                customer_id = f"CUST_{int(cust_clean):05d}"
                            else:
                                total_missing_cust += 1
                                customer_id = f"GUEST_{invoice}"

                            category = categorize_product(desc)
                            product_name = desc.title()
                            revenue = round(price * qty, 2)

                            writer.writerow([
                                invoice,
                                customer_id,
                                dt_formatted,
                                product_name,
                                category,
                                price,
                                qty,
                                revenue,
                                country
                            ])
                            kept_in_sheet += 1
                            total_clean_rows += 1

                print(f"  --> {sheet_name} Kept: {kept_in_sheet:,} rows.")

    print("\n" + "=" * 80)
    print("  DATASET 3 CLEANING AUDIT SUMMARY")
    print("=" * 80)
    print(f"  Total Raw Rows Processed : {total_raw_rows:,}")
    print(f"  Total Cancelled Orders   : {total_cancelled:,}")
    print(f"  Non-positive Qty/Price   : {total_bad_price_qty:,}")
    print(f"  Administrative Codes     : {total_admin:,}")
    print(f"  Missing Descriptions     : {total_missing_desc:,}")
    print(f"  Guest Checkouts Retained : {total_missing_cust:,} (assigned GUEST_<tx_id>)")
    print(f"  Valid Clean Rows Written : {total_clean_rows:,}")
    print(f"  Clean CSV Output         : {OUTPUT_CSV_PATH} ({os.path.getsize(OUTPUT_CSV_PATH)/(1024*1024):.2f} MB)")
    print("=" * 80)

if __name__ == "__main__":
    main()
