"use client";

import { useState } from "react";
import { api } from "@/lib/api";
import { useAppStore } from "@/lib/store/useAppStore";
import { Card, CardHeader, CardTitle, CardDescription, CardContent, CardFooter } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Sparkles, AlertCircle, RefreshCw } from "lucide-react";
import type { Insight } from "@/lib/types";

export default function InsightsPage() {
  const { datasetId } = useAppStore();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [insights, setInsights] = useState<Insight[]>([]);
  const [hasRun, setHasRun] = useState(false);

  const handleGenerate = async () => {
    if (!datasetId) return;
    setLoading(true);
    setError(null);
    try {
      const data = await api.generateInsights(datasetId);
      setInsights(data.insights || []);
      setHasRun(true);
    } catch (err: any) {
      setError(err.message || "Failed to generate insights");
    } finally {
      setLoading(false);
    }
  };

  if (!datasetId) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center text-muted-foreground">
          <Sparkles className="h-12 w-12 mx-auto mb-4 opacity-20" />
          <h2 className="text-xl font-semibold">No Dataset Loaded</h2>
          <p>Please load a dataset in the Data Quality tab first.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6 max-w-7xl">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">AI Insights Center</h1>
          <p className="text-muted-foreground mt-2">Automated statistical insights based on threshold analysis.</p>
        </div>
        <Button onClick={handleGenerate} disabled={loading}>
          {loading ? <RefreshCw className="h-4 w-4 mr-2 animate-spin" /> : <Sparkles className="h-4 w-4 mr-2" />}
          {hasRun ? "Regenerate Insights" : "Generate Insights"}
        </Button>
      </div>

      {error && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>Error</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {hasRun && insights.length === 0 && (
        <Card className="bg-secondary/30 border-dashed">
          <CardContent className="flex flex-col items-center justify-center p-12 text-center text-muted-foreground">
            <Sparkles className="h-10 w-10 mb-4 opacity-30" />
            <p>No significant insights found.</p>
            <p className="text-sm">Try running Pattern Mining or adjusting parameters first.</p>
          </CardContent>
        </Card>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {insights.map((insight, idx) => (
          <Card key={idx} className={`border-l-4 ${insight.importance === 'High' ? 'border-l-indigo-500 shadow-[0_0_15px_rgba(99,102,241,0.05)]' : 'border-l-muted'}`}>
            <CardHeader className="pb-3">
              <div className="flex justify-between items-start">
                <CardDescription className="uppercase tracking-wider text-xs font-semibold text-indigo-400">
                  {insight.type} INSIGHT
                </CardDescription>
                {insight.importance === 'High' && (
                  <span className="text-[10px] uppercase bg-indigo-500/10 text-indigo-400 px-2 py-0.5 rounded-full border border-indigo-500/20">High Impact</span>
                )}
              </div>
              <CardTitle className="text-lg leading-tight pt-2">{insight.pattern}</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="bg-secondary/50 rounded-md p-3 text-sm">
                <div className="font-semibold mb-2 text-xs text-muted-foreground uppercase">Verified Evidence</div>
                <div className="grid grid-cols-2 gap-2">
                  {Object.entries(insight.metrics).map(([key, value]) => (
                    <div key={key}>
                      <span className="block text-[10px] text-muted-foreground uppercase">{key}</span>
                      <span className="font-medium text-foreground">{typeof value === 'number' && key !== 'purchases' ? value.toFixed(4) : String(value)}</span>
                    </div>
                  ))}
                </div>
              </div>
            </CardContent>
            <CardFooter className="pt-0">
               {/* Could add a deep link back to the specific tool here if needed */}
            </CardFooter>
          </Card>
        ))}
      </div>
    </div>
  );
}
