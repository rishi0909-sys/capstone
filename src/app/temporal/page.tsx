"use client";

import { useState } from "react";
import { api } from "@/lib/api";
import { useAppStore } from "@/lib/store/useAppStore";
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { LineChart as LineChartIcon, AlertCircle, Play } from "lucide-react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import type { TemporalTrend } from "@/lib/types";

export default function TemporalPage() {
  const { datasetId } = useAppStore();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [trend, setTrend] = useState<TemporalTrend[] | null>(null);

  const handleAnalyze = async () => {
    if (!datasetId) return;
    setLoading(true);
    setError(null);
    try {
      const data = await api.analyzeTemporal(datasetId);
      if (data.status === "success") {
        setTrend(data.monthly_trend);
      } else {
        setError(data.message || "Analysis unavailable");
      }
    } catch (err: any) {
      setError(err.message || "Failed to run temporal analysis");
    } finally {
      setLoading(false);
    }
  };

  if (!datasetId) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center text-muted-foreground">
          <LineChartIcon className="h-12 w-12 mx-auto mb-4 opacity-20" />
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
          <h1 className="text-3xl font-bold tracking-tight">Temporal Trends</h1>
          <p className="text-muted-foreground mt-2">Analyze demand trends over time.</p>
        </div>
        <Button onClick={handleAnalyze} disabled={loading}>
          {loading ? "Analyzing..." : <><Play className="h-4 w-4 mr-2" /> Run Analysis</>}
        </Button>
      </div>

      {error && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>Notice</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {trend && (
        <Card>
          <CardHeader>
            <CardTitle>Monthly Transaction Trend</CardTitle>
            <CardDescription>Overall transaction volume over time.</CardDescription>
          </CardHeader>
          <CardContent className="h-[500px]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={trend} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                <XAxis dataKey="year_month" stroke="#888" />
                <YAxis stroke="#888" />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#1a1a1a', borderColor: '#333' }}
                />
                <Line type="monotone" dataKey="value" stroke="#6366f1" strokeWidth={3} dot={{ r: 4, fill: '#6366f1' }} activeDot={{ r: 8 }} />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
