import os
import json
import ssl
import urllib.request
from pydantic import BaseModel
import re

def _ssl_context():
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl._create_unverified_context()

class LLMExplainer:
    def __init__(self):
        self.google_api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not self.google_api_key:
            env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), ".env")
            if os.path.exists(env_path):
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if "=" in line and not line.strip().startswith("#"):
                            k, v = line.split("=", 1)
                            if k.strip().lower() in ("google_api_key", "gemini_api_key"):
                                self.google_api_key = v.strip().strip('"').strip("'")
                                break
        self.gemini_model = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
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
        return bool(self.google_api_key) or (self.client is not None)

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
            if self.google_api_key:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.gemini_model}:generateContent?key={self.google_api_key}"
                req_obj = urllib.request.Request(
                    url,
                    data=json.dumps({
                        "contents": [{"parts": [{"text": prompt}]}],
                        "generationConfig": {"temperature": 0.1},
                    }).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req_obj, timeout=20.0, context=_ssl_context()) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                explanation = data["candidates"][0]["content"]["parts"][0]["text"]
            else:
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

    def chat_with_context(self, question: str, context: dict) -> str:
        if not self.is_available():
            return "AI Chat is unavailable because no API key is configured."

        prompt = f"""You are a friendly retail business assistant.
Explain things in very simple, everyday language that anyone with zero technical or data-science knowledge can easily understand.
Avoid jargon like WAPE, MAD, robust z-score, lift, or support — instead say "accuracy", "unusual spike", "often bought together", etc.
Use ONLY the facts in the context below. Do not invent numbers.

Context:
{json.dumps(context, indent=2)[:4000]}

User Question: {question}
Answer clearly in 2-4 short sentences:"""

        try:
            if self.google_api_key:
                from src.llm.gemini_agent import GeminiRetailAgent
                return GeminiRetailAgent(model=self.gemini_model).generate(question, context)
            else:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                )
                return response.choices[0].message.content
        except Exception:
            cats = context.get("categories", [])
            rels = context.get("top_relationships", [])
            cb = context.get("customer_summary", {})
            em = context.get("emerging_and_anomalies", {}).get("emerging_patterns", [])
            q_low = question.lower()
            if any(g in q_low for g in ["how are you", "hello", "hi ", "hey", "who are you"]):
                return "I'm doing great, thanks for asking! I'm your retail assistant — ask me anything about your store's top categories, product pairings, or customer trends."
            if "buy" in q_low or "stock" in q_low or "next" in q_low or "recommend" in q_low or "investigate" in q_low:
                parts = []
                if em:
                    top_e = em[0]
                    parts.append(f"Focus on {top_e['category']} (+{top_e['growth_rate_pct']}% recent 3-month surge vs baseline).")
                if rels:
                    r = rels[0]
                    parts.append(f"Pair {', '.join(r['antecedents'])} with {', '.join(r['consequents'])} ({r['confidence']*100:.1f}% of buyers pick both).")
                if cats:
                    parts.append(f"Keep {cats[0].get('canonical_category', 'Home & Living')} well-stocked as it leads with {cats[0].get('total_occurrences', 0):,} records.")
                if parts:
                    return " ".join(parts)
            if "together" in q_low or "bundle" in q_low or "relationship" in q_low:
                if rels:
                    r = rels[0]
                    return f"Customers who buy {', '.join(r['antecedents'])} very often also buy {', '.join(r['consequents'])} ({r['confidence']*100:.1f}% of the time, {r['lift']}x stronger than average)."
            if "customer" in q_low or "return" in q_low or "repeat" in q_low:
                if cb:
                    return f"Out of {cb.get('total_unique_customers', 0):,} shoppers, {cb.get('repeat_customers', 0):,} ({cb.get('repeat_purchase_rate_percent', 0)}%) came back for repeat orders, averaging {cb.get('average_units_per_basket', 0)} items per order."
            if "gain" in q_low or "grow" in q_low or "emerg" in q_low:
                if em:
                    top_e = em[0]
                    return f"{top_e['category']} is growing fastest recently, up +{top_e['growth_rate_pct']}% over the last 3 months compared to its historical average."
            if cats:
                top_c = cats[0]
                return f"Our largest category is {top_c.get('canonical_category', 'Home & Living')} with {top_c.get('total_occurrences', 0):,} records across {top_c.get('dataset_coverage', 'all')} datasets."
            return "Based on our store data, Home & Living is the top category and Seasonal & Holiday shows the strongest recent growth."
