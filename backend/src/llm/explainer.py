import os
import json
from pydantic import BaseModel
import re

class LLMExplainer:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("OPENAI_BASE_URL", "").strip()
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o")
        
        if not self.base_url:
            self.base_url = None # Use default OpenAI
            
        if self.api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url
                )
            except Exception as e:
                print(f"[Warning] Failed to initialize OpenAI client: {e}")
                self.client = None
        else:
            self.client = None

    def is_available(self):
        return self.client is not None

    def explain_association_rule(self, rule_data: dict):
        if not self.is_available():
            return {"error": "LLM API Key not provided. Explanation disabled."}
            
        prompt = f"""
        You are an Explainable AI analytics assistant.
        
        Explain the following statistical association rule in plain English.
        Use ONLY the supplied values. Do not invent statistics. Do not modify numerical values.
        Do not claim causation. Clearly distinguish observed association from causal inference.
        
        Data:
        Antecedent (If they buy): {', '.join(rule_data['antecedents'])}
        Consequent (Then they also buy): {', '.join(rule_data['consequents'])}
        Support: {rule_data['support']:.4f}
        Confidence: {rule_data['confidence']:.4f}
        Lift: {rule_data['lift']:.4f}
        
        Format your response with the following headers:
        ### What was discovered
        ### Why it matters
        ### Possible business action
        ### Evidence
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a precise data science explainer."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            explanation = response.choices[0].message.content
            
            # Hallucination Control: Evidence Validation
            # Verify that the numbers in the prompt are present in the output
            metrics = [f"{rule_data['support']:.4f}", f"{rule_data['confidence']:.4f}", f"{rule_data['lift']:.4f}"]
            
            # Simple validation: just checking if the numbers are present in the text
            for metric in metrics:
                if metric not in explanation:
                     return {"error": f"Generated explanation failed evidence validation (Missing metric {metric}).", "raw_output": explanation}
            
            return {"explanation": explanation}
            
        except Exception as e:
            return {"error": f"LLM API Error: {str(e)}"}
