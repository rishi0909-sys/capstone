import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder

class FPGrowthMiner:
    def __init__(self, basket_df: pd.DataFrame):
        """
        basket_df should have a column 'product_name' which contains lists of products per transaction.
        """
        self.basket_df = basket_df
        self.frequent_itemsets = None
        self.rules = None
        self.te = TransactionEncoder()
        
    def _create_sparse_matrix(self):
        # Convert lists of items into a sparse boolean matrix
        transactions = self.basket_df['product_name'].tolist()
        te_ary = self.te.fit(transactions).transform(transactions, sparse=True)
        # Using sparse format dataframe
        sparse_df = pd.DataFrame.sparse.from_spmatrix(te_ary, columns=self.te.columns_)
        return sparse_df

    def mine_patterns(self, min_support=0.01, min_confidence=0.30, min_lift=1.10, 
                      max_antecedent_size=None, max_consequent_size=None):
        
        if self.basket_df is None or self.basket_df.empty:
            return pd.DataFrame(), pd.DataFrame()
            
        sparse_df = self._create_sparse_matrix()
        
        # Mine frequent itemsets
        try:
            self.frequent_itemsets = fpgrowth(sparse_df, min_support=min_support, use_colnames=True)
        except Exception as e:
            # Handle potential memory/timeout errors gracefully
            print(f"FP-Growth failed: {str(e)}")
            return pd.DataFrame(), pd.DataFrame()

        if self.frequent_itemsets.empty:
            return self.frequent_itemsets, pd.DataFrame()

        # Generate association rules
        try:
            self.rules = association_rules(self.frequent_itemsets, metric="confidence", min_threshold=min_confidence)
        except Exception as e:
            print(f"Rule generation failed: {str(e)}")
            return self.frequent_itemsets, pd.DataFrame()

        if self.rules.empty:
            return self.frequent_itemsets, self.rules

        # Filter by lift
        self.rules = self.rules[self.rules['lift'] >= min_lift]
        
        # Calculate lengths
        self.rules['antecedent_len'] = self.rules['antecedents'].apply(lambda x: len(x))
        self.rules['consequent_len'] = self.rules['consequents'].apply(lambda x: len(x))
        
        # Filter by max sizes if provided
        if max_antecedent_size is not None:
            self.rules = self.rules[self.rules['antecedent_len'] <= max_antecedent_size]
        if max_consequent_size is not None:
            self.rules = self.rules[self.rules['consequent_len'] <= max_consequent_size]
            
        # Clean up sets for easier display/JSON serialization
        self.rules['antecedents'] = self.rules['antecedents'].apply(lambda x: list(x))
        self.rules['consequents'] = self.rules['consequents'].apply(lambda x: list(x))
        self.frequent_itemsets['itemsets'] = self.frequent_itemsets['itemsets'].apply(lambda x: list(x))

        return self.frequent_itemsets, self.rules
