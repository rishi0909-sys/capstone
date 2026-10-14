"use client";

import { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { apiClient } from "@/lib/api/client";
import { useAppStore } from "@/lib/store/useAppStore";
import { MiningRequest, MiningResponse, RuleItem, ExplainRequest, ExplainResponse } from "@/lib/types";
import { ErrorState } from "@/components/shared/ErrorState";
import { Button } from "@/components/ui/button";
import { Network, Search, ArrowRight, Sparkles, AlertCircle } from "lucide-react";

import { motion, AnimatePresence } from "framer-motion";

export default function PatternsPage() {
  return <PatternsContent />;
}

function PatternsContent() {
  const { activeCategory } = useAppStore();
  const [activeRule, setActiveRule] = useState<RuleItem | null>(null);
  const [exploreQuery, setExploreQuery] = useState("");

  // MOCKED DATA: Replace this when the backend is ready
  const miningData = {
    rule_count: 3,
    rules: [
      { antecedent: ["Gaming Laptop"], consequent: ["Gaming Mouse", "Mousepad"], support: 0.15, confidence: 0.85, lift: 3.2 },
      { antecedent: ["Mechanical Keyboard"], consequent: ["Wrist Rest"], support: 0.08, confidence: 0.72, lift: 2.5 },
      { antecedent: ["4K Monitor"], consequent: ["DisplayPort Cable"], support: 0.12, confidence: 0.91, lift: 4.1 },
    ]
  };
  const isLoadingRules = false;
  const miningError = null;

  if (isLoadingRules) {
    return (
      <div className="flex flex-col items-center justify-center h-64 text-muted-foreground">
         <Network className="h-8 w-8 mb-4 animate-pulse opacity-50" />
         <p>Loading precomputed product patterns...</p>
      </div>
    );
  }

  if (miningError) {
    return <ErrorState title="Pattern Discovery Failed" message="Error loading mock data" />;
  }

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-end">
        <div>
          <h1 className="text-3xl font-bold flex items-center">
            <Network className="mr-3 h-8 w-8 text-primary" />
            What Products Go Together?
          </h1>
          <p className="text-muted-foreground mt-2">
            Explore {miningData?.rule_count || 0} purchasing relationships discovered in {activeCategory || 'your dataset'}.
          </p>
        </div>
      </div>

      <div className="bg-card border rounded-lg p-6 shadow-sm mb-6">
        <h2 className="text-lg font-semibold mb-2">What do you want to explore?</h2>
        <div className="relative max-w-xl">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <input 
            type="text" 
            placeholder="e.g. 'Show me what people buy with Laptops'" 
            value={exploreQuery}
            onChange={(e) => setExploreQuery(e.target.value)}
            className="w-full bg-background border rounded-md pl-10 pr-4 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
          />
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="lg:col-span-2 border rounded-lg bg-card overflow-hidden shadow-lg shadow-black/5"
        >
           <div className="p-4 border-b flex justify-between items-center bg-muted/20">
              <h3 className="font-semibold">Products Often Bought Together</h3>
           </div>
           {miningData?.rules && miningData.rules.length > 0 ? (
             <div className="overflow-x-auto">
               <table className="w-full text-sm text-left">
                 <thead className="text-xs text-muted-foreground uppercase bg-muted/50">
                   <tr>
                     <th className="px-4 py-3">Antecedent</th>
                     <th className="px-4 py-3">Consequent</th>
                     <th className="px-4 py-3 text-right">Support</th>
                     <th className="px-4 py-3 text-right">Confidence</th>
                     <th className="px-4 py-3 text-right">Lift</th>
                     <th className="px-4 py-3"></th>
                   </tr>
                 </thead>
                 <motion.tbody
                    initial="hidden"
                    animate="visible"
                    variants={{
                      hidden: { opacity: 0 },
                      visible: { opacity: 1, transition: { staggerChildren: 0.05 } }
                    }}
                 >
                   {miningData.rules.map((rule, idx) => (
                     <motion.tr 
                       variants={{
                         hidden: { opacity: 0, x: -10 },
                         visible: { opacity: 1, x: 0 }
                       }}
                       key={idx} 
                       className={`border-b transition-colors cursor-pointer hover:bg-muted/30 ${activeRule === rule ? 'bg-primary/5 border-l-2 border-l-indigo-500' : 'border-l-2 border-l-transparent'}`}
                       onClick={() => setActiveRule(rule)}
                     >
                       <td className="px-4 py-3 max-w-[200px] truncate" title={rule.antecedent.join(', ')}>
                         {rule.antecedent.join(', ')}
                       </td>
                       <td className="px-4 py-3 max-w-[200px] truncate" title={rule.consequent.join(', ')}>
                         {rule.consequent.join(', ')}
                       </td>
                       <td className="px-4 py-3 text-right font-mono">{(rule.support).toFixed(4)}</td>
                       <td className="px-4 py-3 text-right font-mono">{(rule.confidence).toFixed(2)}</td>
                       <td className="px-4 py-3 text-right font-mono">{(rule.lift).toFixed(2)}</td>
                       <td className="px-4 py-3 text-right">
                         <Button variant="ghost" size="sm" onClick={(e) => { e.stopPropagation(); setActiveRule(rule); }}>
                           View
                         </Button>
                       </td>
                     </motion.tr>
                   ))}
                 </motion.tbody>
               </table>
             </div>
           ) : (
             <div className="p-8 text-center text-muted-foreground">
               No patterns met the minimum thresholds.
             </div>
           )}
        </motion.div>

        <div className="lg:col-span-1">
          <AnimatePresence mode="wait">
            {activeRule ? (
              <motion.div
                key="detail"
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                transition={{ duration: 0.3 }}
                className="h-full"
              >
                <PatternDetail rule={activeRule} />
              </motion.div>
            ) : (
              <motion.div 
                key="empty"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="border rounded-lg p-8 bg-card/50 flex flex-col items-center justify-center text-center h-full text-muted-foreground shadow-sm"
              >
                <Search className="h-8 w-8 mb-4 opacity-50" />
                <p>Select a rule from the table to view details and generate AI insights.</p>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </div>
  );
}

function PatternDetail({ rule }: { rule: RuleItem }) {
  const explainMutation = useMutation({
    mutationFn: async (req: ExplainRequest) => {
      return apiClient.post<ExplainResponse>("/api/llm/explain", req);
    },
  });

  const handleExplain = () => {
    explainMutation.mutate({
      antecedent: rule.antecedent,
      consequent: rule.consequent,
      support: rule.support,
      confidence: rule.confidence,
      lift: rule.lift,
    });
  };

  return (
    <div className="border rounded-lg bg-card overflow-hidden sticky top-6">
      <div className="p-4 border-b bg-muted/20">
        <h3 className="font-semibold">Discovery Details</h3>
      </div>
      
      <div className="p-6 space-y-6">
        <div className="space-y-4 bg-muted/30 p-4 rounded-md">
          <div>
            <span className="text-xs text-muted-foreground uppercase tracking-wider font-semibold block mb-1">If customer buys:</span>
            <div className="font-medium text-lg leading-tight">{rule.antecedent.join(', ')}</div>
          </div>
          
          <div className="flex justify-center">
            <ArrowRight className="text-muted-foreground" />
          </div>
          
          <div>
            <span className="text-xs text-muted-foreground uppercase tracking-wider font-semibold block mb-1">They also buy:</span>
            <div className="font-medium text-lg leading-tight text-primary">{rule.consequent.join(', ')}</div>
          </div>
        </div>

        <div>
          <h4 className="text-sm font-semibold mb-3">Statistical Evidence</h4>
          <div className="grid grid-cols-3 gap-2 text-center">
            <div className="bg-secondary/50 p-2 rounded">
              <div className="text-xs text-muted-foreground">Support</div>
              <div className="font-mono text-sm">{(rule.support).toFixed(4)}</div>
            </div>
            <div className="bg-secondary/50 p-2 rounded">
              <div className="text-xs text-muted-foreground">Confidence</div>
              <div className="font-mono text-sm">{(rule.confidence).toFixed(2)}</div>
            </div>
            <div className="bg-secondary/50 p-2 rounded">
              <div className="text-xs text-muted-foreground">Lift</div>
              <div className="font-mono text-sm">{(rule.lift).toFixed(2)}</div>
            </div>
          </div>
        </div>

        <div className="pt-4 border-t">
           <div className="flex items-center justify-between mb-4">
             <h4 className="text-sm font-semibold">Help Me Understand This</h4>
             {!explainMutation.data && !explainMutation.isPending && (
               <Button size="sm" onClick={handleExplain} variant="outline" className="h-8">
                 <Sparkles className="mr-2 h-3 w-3 text-indigo-500" />
                 Generate
               </Button>
             )}
           </div>

           {explainMutation.isPending && (
             <div className="text-sm text-muted-foreground animate-pulse flex items-center">
               <Sparkles className="mr-2 h-4 w-4" /> Generating natural language explanation...
             </div>
           )}

           {explainMutation.isError && (
             <div className="text-sm text-destructive flex items-start bg-destructive/10 p-3 rounded">
               <AlertCircle className="mr-2 h-4 w-4 mt-0.5 shrink-0" />
               <span>{explainMutation.error.message}</span>
             </div>
           )}

           {explainMutation.data && (
             <div className="text-sm leading-relaxed bg-indigo-500/10 dark:bg-indigo-500/5 p-4 rounded-md border border-indigo-500/20">
                {explainMutation.data.error ? (
                  <span className="text-destructive">{explainMutation.data.error}</span>
                ) : (
                  explainMutation.data.explanation
                )}
             </div>
           )}
        </div>
      </div>
    </div>
  );
}
