import pandas as pd
import numpy as np

class TemporalAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        
    def _extract_features(self):
        if 'date' not in self.df.columns:
            return False
            
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df['year'] = self.df['date'].dt.year
        self.df['month'] = self.df['date'].dt.month
        self.df['week'] = self.df['date'].dt.isocalendar().week
        self.df['day'] = self.df['date'].dt.day
        self.df['day_of_week'] = self.df['date'].dt.dayofweek
        self.df['quarter'] = self.df['date'].dt.quarter
        self.df['year_month'] = self.df['date'].dt.to_period('M').astype(str)
        self.df['year_week'] = self.df['date'].dt.strftime('%Y-W%V')
        
        return True
        
    def analyze_trends(self, granularity='month', metric='transaction_count', target_product=None, target_category=None):
        """
        granularity: 'month', 'week', 'day'
        metric: 'transaction_count', 'quantity', 'revenue'
        """
        if not self._extract_features():
            return None
            
        filtered_df = self.df
        if target_product:
            filtered_df = filtered_df[filtered_df['product_name'] == target_product]
        if target_category and 'category' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['category'] == target_category]
            
        if filtered_df.empty:
            return None
            
        group_col = f"year_{granularity}" if granularity in ['month', 'week'] else 'date'
        if group_col == 'date':
            filtered_df['date_only'] = filtered_df['date'].dt.date
            group_col = 'date_only'
            
        if metric == 'transaction_count':
            trend = filtered_df.groupby(group_col)['transaction_id'].nunique().reset_index(name='value')
        elif metric == 'quantity' and 'quantity' in filtered_df.columns:
            trend = filtered_df.groupby(group_col)['quantity'].sum().reset_index(name='value')
        elif metric == 'revenue' and 'price' in filtered_df.columns and 'quantity' in filtered_df.columns:
            filtered_df['rev'] = filtered_df['price'] * filtered_df['quantity']
            trend = filtered_df.groupby(group_col)['rev'].sum().reset_index(name='value')
        else:
            trend = filtered_df.groupby(group_col).size().reset_index(name='value')
            
        trend = trend.sort_values(by=group_col)
        return trend

    def analyze_seasonality(self):
        """Aggregate by month across all years to find seasonal patterns."""
        if not self._extract_features():
            return None
            
        # Overall transactions by month
        seasonal = self.df.groupby('month')['transaction_id'].nunique().reset_index(name='transactions')
        
        # We can also do it per product
        prod_seasonal = self.df.groupby(['product_name', 'month']).size().reset_index(name='purchases')
        
        return seasonal, prod_seasonal
