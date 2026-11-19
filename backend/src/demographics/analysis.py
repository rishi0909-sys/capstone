"""
Customer Behavior & Demographic Analyzer
========================================
Analyzes customer purchasing preferences, basket profiles, and repeat-buyer behavior.
Supports authentic customer segments (RFM, basket size, region) without fabricating ages.
"""

import pandas as pd
import numpy as np

class DemographicAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        
    def _create_age_groups(self):
        if 'age' not in self.df.columns or self.df['age'].isna().all():
            return False
            
        bins = [0, 25, 35, 45, 55, 100]
        labels = ['18-25', '26-35', '36-45', '46-55', '56+']
        self.df['age_group'] = pd.cut(self.df['age'], bins=bins, labels=labels, right=True)
        return True

    def _create_behavioral_segments(self):
        """
        Derives authentic customer behavioral segments when demographic age is absent:
        - Customer Frequency: 'One-Time Buyer', 'Repeat Shopper', 'Loyal / VIP'
        - Basket Size Tier: 'Small (1-2 items)', 'Medium (3-9 items)', 'Bulk / Wholesale (10+ items)'
        """
        if 'customer_id' not in self.df.columns:
            return False

        # Frequency tier per customer
        cust_orders = self.df.groupby('customer_id')['transaction_id'].nunique()
        def get_freq_tier(n_orders):
            if n_orders == 1:
                return 'One-Time Buyer'
            elif n_orders <= 4:
                return 'Repeat Shopper'
            else:
                return 'Loyal / VIP'

        cust_freq_map = cust_orders.apply(get_freq_tier).to_dict()
        self.df['customer_segment'] = self.df['customer_id'].map(cust_freq_map).fillna('Guest Buyer')
        return True

    def analyze_category_preferences(self):
        """
        Analyzes category preference by customer segment (or age group if present).
        """
        has_age = self._create_age_groups()
        has_behavior = self._create_behavioral_segments()

        if not has_age and not has_behavior:
            return None

        if 'category' not in self.df.columns:
            return None

        group_col = 'age_group' if has_age else 'customer_segment'
        
        pref = self.df.groupby([group_col, 'category']).size().reset_index(name='purchase_count')
        total_by_seg = self.df.groupby(group_col).size().reset_index(name='total_segment_purchases')
        pref = pd.merge(pref, total_by_seg, on=group_col)
        pref['probability'] = pref['purchase_count'] / pref['total_segment_purchases']
        
        # Ensure consistent column name for downstream consumers
        if not has_age:
            pref['age_group'] = pref['customer_segment']

        return pref

    def analyze_product_preferences(self):
        """Analyze product preference by segment."""
        has_age = self._create_age_groups()
        has_behavior = self._create_behavioral_segments()

        if not has_age and not has_behavior:
            return None

        if 'product_name' not in self.df.columns:
            return None

        group_col = 'age_group' if has_age else 'customer_segment'
        pref = self.df.groupby([group_col, 'product_name']).size().reset_index(name='purchase_count')
        total_by_seg = self.df.groupby(group_col).size().reset_index(name='total_segment_purchases')
        pref = pd.merge(pref, total_by_seg, on=group_col)
        pref['probability'] = pref['purchase_count'] / pref['total_segment_purchases']
        
        if not has_age:
            pref['age_group'] = pref['customer_segment']

        return pref

    def get_summary(self):
        has_age = self._create_age_groups()
        has_behavior = self._create_behavioral_segments()

        if has_age:
            return {
                "status": "Success",
                "segmentation_type": "Demographic Age",
                "age_groups": self.df['age_group'].value_counts().to_dict(),
                "genders": self.df['gender'].value_counts().to_dict() if 'gender' in self.df.columns else {}
            }
        elif has_behavior:
            # Legitimate customer analytics
            cust_df = self.df[self.df['customer_id'] != 'UNREGISTERED_GUEST']
            total_registered = cust_df['customer_id'].nunique()
            orders_per_cust = cust_df.groupby('customer_id')['transaction_id'].nunique()
            repeat_customers = int((orders_per_cust > 1).sum())
            repeat_rate = round((repeat_customers / total_registered * 100), 1) if total_registered > 0 else 0.0

            # Items per basket
            basket_sizes = self.df.groupby('transaction_id')['quantity'].sum()
            avg_basket_size = round(float(basket_sizes.mean()), 1)

            return {
                "status": "Success",
                "segmentation_type": "Behavioral & Frequency Tiers",
                "total_unique_customers": total_registered,
                "repeat_customers": repeat_customers,
                "repeat_purchase_rate_percent": repeat_rate,
                "average_units_per_basket": avg_basket_size,
                "segments": self.df['customer_segment'].value_counts().to_dict(),
                "countries": self.df['country'].value_counts().head(5).to_dict() if 'country' in self.df.columns else {}
            }
        else:
            return {"status": "No segmentation data available"}
