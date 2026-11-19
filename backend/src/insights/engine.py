import pandas as pd

class InsightEngine:
    def __init__(self):
        self.insights = []
        
    def generate_association_insights(self, rules_df: pd.DataFrame, top_n=5, min_lift=1.5):
        if rules_df is None or rules_df.empty:
            return
            
        high_lift = rules_df[rules_df['lift'] >= min_lift].sort_values(by='lift', ascending=False).head(top_n)
        
        for _, row in high_lift.iterrows():
            ant = ", ".join(row['antecedents'])
            con = ", ".join(row['consequents'])
            self.insights.append({
                "type": "Association",
                "pattern": f"Strong association between {ant} and {con}",
                "metrics": {
                    "support": round(row['support'], 4),
                    "confidence": round(row['confidence'], 4),
                    "lift": round(row['lift'], 4)
                },
                "raw_data": row.to_dict(),
                "importance": "High" if row['lift'] > 2.0 else "Medium"
            })
            
    def generate_demographic_insights(self, demographic_pref_df: pd.DataFrame, threshold=0.3):
        if demographic_pref_df is None or demographic_pref_df.empty:
            return
            
        # Find demographic segments with unusually high probability for a category
        high_prob = demographic_pref_df[demographic_pref_df['probability'] >= threshold]
        
        for _, row in high_prob.iterrows():
            if 'category' in row:
                target = row['category']
            elif 'product_name' in row:
                target = row['product_name']
            else:
                continue
                
            self.insights.append({
                "type": "Demographic",
                "pattern": f"Age group {row['age_group']} has strong preference for {target}",
                "metrics": {
                    "probability": round(row['probability'], 4),
                    "purchases": int(row['purchase_count'])
                },
                "raw_data": row.to_dict(),
                "importance": "Medium"
            })

    def get_insights(self):
        return self.insights
