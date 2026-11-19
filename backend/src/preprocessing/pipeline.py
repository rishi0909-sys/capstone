import pandas as pd
import numpy as np

class PreprocessingPipeline:
    def __init__(self, df: pd.DataFrame, column_mapping: dict = None):
        self.raw_df = df.copy()
        self.df = df.copy()
        self.column_mapping = column_mapping or {}
        self.summary = {}
        
    def _map_columns(self):
        """Map standard columns based on user input or auto-detect."""
        standard_cols = ['transaction_id', 'product_name', 'date', 'customer_id', 'age', 'gender', 'category', 'price', 'quantity']
        
        # Auto-detect if mapping not provided
        if not self.column_mapping:
            cols = [c.lower() for c in self.df.columns]
            for std_col in standard_cols:
                # Direct match
                if std_col in cols:
                    actual_col = self.df.columns[cols.index(std_col)]
                    self.column_mapping[std_col] = actual_col
                else:
                    # Partial match heuristics
                    for c in cols:
                        if std_col.replace('_', '') in c.replace('_', ''):
                            actual_col = self.df.columns[cols.index(c)]
                            self.column_mapping[std_col] = actual_col
                            break
                            
        # Rename columns to standard names internally
        rename_dict = {v: k for k, v in self.column_mapping.items() if k in standard_cols and v in self.df.columns}
        self.df = self.df.rename(columns=rename_dict)
        
    def validate_required(self) -> tuple[bool, str]:
        self._map_columns()
        if 'transaction_id' not in self.df.columns or 'product_name' not in self.df.columns:
            return False, "Dataset must contain at least 'transaction_id' and 'product_name' columns."
        return True, ""
        
    def clean_data(self):
        self.summary['initial_rows'] = len(self.raw_df)
        
        # Drop duplicates
        initial_len = len(self.df)
        self.df = self.df.drop_duplicates()
        self.summary['duplicate_rows_removed'] = initial_len - len(self.df)
        
        # Handle missing values in required columns
        missing_before = len(self.df)
        self.df = self.df.dropna(subset=['transaction_id', 'product_name'])
        self.summary['rows_removed_missing_critical'] = missing_before - len(self.df)
        
        # Parse Dates
        if 'date' in self.df.columns:
            self.df['date'] = pd.to_datetime(self.df['date'], errors='coerce')
            # drop invalid dates
            invalid_dates = self.df['date'].isna().sum()
            self.df = self.df.dropna(subset=['date'])
            self.summary['invalid_dates_removed'] = int(invalid_dates)
            self.summary['date_range'] = f"{self.df['date'].min().strftime('%Y-%m-%d')} to {self.df['date'].max().strftime('%Y-%m-%d')}"
            
        # Clean product names
        self.df['product_name'] = self.df['product_name'].astype(str).str.strip().str.title()
        
        # Datatypes
        if 'price' in self.df.columns:
            self.df['price'] = pd.to_numeric(self.df['price'], errors='coerce').fillna(0.0)
        if 'quantity' in self.df.columns:
            self.df['quantity'] = pd.to_numeric(self.df['quantity'], errors='coerce').fillna(1)
        if 'age' in self.df.columns:
            self.df['age'] = pd.to_numeric(self.df['age'], errors='coerce')
            
        self.summary['final_rows'] = len(self.df)
        self.summary['unique_transactions'] = self.df['transaction_id'].nunique()
        self.summary['unique_products'] = self.df['product_name'].nunique()
        
        # Compute missing values summary per column
        self.summary['missing_values'] = self.df.isna().sum().to_dict()
        
        return self.df
        
    def get_basket_df(self):
        """Groups by transaction_id and returns a list of products per transaction."""
        if 'transaction_id' not in self.df.columns or 'product_name' not in self.df.columns:
            return None
        return self.df.groupby('transaction_id')['product_name'].apply(lambda x: list(set(x))).reset_index()
        
    def get_summary(self):
        return self.summary
