"use client";

import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { apiClient } from "@/lib/api/client";
import { useAppStore } from "@/lib/store/useAppStore";
import { ForecastRequest, ForecastResponse } from "@/lib/types";
import { ErrorState } from "@/components/shared/ErrorState";
import { Button } from "@/components/ui/button";
import { TrendingUp, Settings } from "lucide-react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine
} from "recharts";

export default function ForecastingPage() {
  return <ForecastingContent />;
}

function ForecastingContent() {
  const { datasetId } = useAppStore();
  
  // Forecast configuration state
  const [targetProduct, setTargetProduct] = useState<string>("Overall");
  const [horizon, setHorizon] = useState<number>(4);
  const [granularity, setGranularity] = useState<string>("W"); // 'W'eekly, 'M'onthly

  const forecastMutation = useMutation({
    mutationFn: async (req: ForecastRequest) => {
      return apiClient.post<ForecastResponse>("/api/forecast/run", req);
    }
  });

  const handleRunForecast = () => {
    forecastMutation.mutate({
      dataset_id: datasetId || "demo_retail_data",
      target_product: targetProduct,
      horizon,
      granularity
    });
  };

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-end">
        <div>
          <h1 className="text-3xl font-bold flex items-center">
            <TrendingUp className="mr-3 h-8 w-8 text-primary" />
            What Could Demand Look Like Next?
          </h1>
          <p className="text-muted-foreground mt-2">
            Explore future demand projections modeled by XGBoost.
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        
        {/* Configuration Sidebar */}
        <div className="lg:col-span-1 border rounded-lg bg-card p-4 space-y-6 self-start">
          <div className="flex items-center font-semibold border-b pb-2">
             <Settings className="mr-2 h-4 w-4" /> Configuration
          </div>
          
          <div className="space-y-2">
            <label className="text-sm font-medium">Target Product</label>
            <select 
              value={targetProduct} 
              onChange={(e) => setTargetProduct(e.target.value)}
              className="w-full flex h-10 w-full items-center justify-between rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
            >
              <option value="Overall">Overall (All Products)</option>
              <option value="Laptop">Laptop</option>
              <option value="Monitor">Monitor</option>
              <option value="Mouse">Mouse</option>
              <option value="Keyboard">Keyboard</option>
              <option value="Headphones">Headphones</option>
            </select>
          </div>
          
          <div className="space-y-2">
            <label className="text-sm font-medium">Forecast Horizon</label>
            <input 
              type="number"
              min={1}
              max={12}
              value={horizon}
              onChange={(e) => setHorizon(parseInt(e.target.value))}
              className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
            />
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium">Granularity</label>
            <select 
              value={granularity} 
              onChange={(e) => setGranularity(e.target.value)}
              className="w-full flex h-10 w-full items-center justify-between rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2"
            >
              <option value="D">Daily</option>
              <option value="W">Weekly</option>
              <option value="M">Monthly</option>
            </select>
          </div>

          <Button 
            className="w-full" 
            onClick={handleRunForecast}
            disabled={forecastMutation.isPending}
          >
            {forecastMutation.isPending ? "Loading..." : "View Forecast"}
          </Button>
        </div>

        {/* Results Area */}
        <div className="lg:col-span-3 border rounded-lg bg-card p-6 min-h-[500px] flex flex-col">
          {forecastMutation.isPending ? (
             <div className="flex flex-col items-center justify-center flex-1 text-muted-foreground h-full">
                <TrendingUp className="h-12 w-12 mb-4 animate-pulse opacity-50" />
                <p>Loading precomputed predictions...</p>
             </div>
          ) : forecastMutation.isError ? (
             <ErrorState 
               title="Forecasting Failed" 
               message={forecastMutation.error.message}
               onRetry={handleRunForecast}
             />
          ) : forecastMutation.data ? (
             <ForecastResults data={forecastMutation.data} />
          ) : (
             <div className="flex flex-col items-center justify-center flex-1 text-muted-foreground h-full min-h-[400px]">
                <TrendingUp className="h-12 w-12 mb-4 opacity-20" />
                <p>Select a product to view its demand forecast.</p>
             </div>
          )}
        </div>
      </div>
    </div>
  );
}

import { motion, Variants } from "framer-motion";

function ForecastResults({ data }: { data: ForecastResponse }) {
  // Transform backend arrays into unified array of objects for Recharts
  const chartData: any[] = [];
  
  // 1. Add historical actuals
  data.historical_dates.forEach((date, i) => {
    chartData.push({
      date,
      historical: data.historical_actuals[i],
      testPrediction: null,
      futurePrediction: null
    });
  });

  // 2. Add test predictions
  data.test_dates.forEach((date, i) => {
    const existingEntry = chartData.find(d => d.date === date);
    if (existingEntry) {
      existingEntry.historical = data.test_actuals[i];
      existingEntry.testPrediction = data.test_preds[i];
    } else {
      chartData.push({
        date,
        historical: data.test_actuals[i],
        testPrediction: data.test_preds[i],
        futurePrediction: null
      });
    }
  });

  // 3. Add future predictions
  data.future_dates.forEach((date, i) => {
    chartData.push({
      date,
      historical: null,
      testPrediction: null,
      futurePrediction: data.future_preds[i]
    });
  });

  const container: Variants = {
    hidden: { opacity: 0 },
    show: {
      opacity: 1,
      transition: { staggerChildren: 0.1 }
    }
  };

  const item: Variants = {
    hidden: { opacity: 0, y: 20 },
    show: { opacity: 1, y: 0, transition: { type: "spring", stiffness: 300, damping: 24 } }
  };

  return (
    <motion.div 
      variants={container}
      initial="hidden"
      animate="show"
      className="space-y-8 flex-1 flex flex-col"
    >
      <motion.div variants={item}>
        <h3 className="font-semibold text-lg mb-4">Volume Prediction</h3>
        <div className="h-[350px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
              <XAxis dataKey="date" tick={{fontSize: 12}} />
              <YAxis tick={{fontSize: 12}} />
              <Tooltip 
                contentStyle={{ backgroundColor: 'hsl(var(--card))', borderColor: 'hsl(var(--border))' }}
                itemStyle={{ color: 'hsl(var(--foreground))' }}
              />
              <Legend />
              <Line 
                type="monotone" 
                dataKey="historical" 
                name="Actual Data" 
                stroke="#8884d8" 
                dot={false}
                strokeWidth={2}
              />
              <Line 
                type="monotone" 
                dataKey="testPrediction" 
                name="Test Prediction" 
                stroke="#82ca9d" 
                strokeDasharray="5 5"
                dot={false}
                strokeWidth={2}
              />
              <Line 
                type="monotone" 
                dataKey="futurePrediction" 
                name="Future Forecast" 
                stroke="#ffc658" 
                dot={{ r: 4 }}
                strokeWidth={2}
              />
              {data.future_dates.length > 0 && (
                <ReferenceLine 
                  x={data.future_dates[0]} 
                  stroke="hsl(var(--muted-foreground))" 
                  strokeDasharray="3 3" 
                  label={{ position: 'top', value: 'Future', fill: 'hsl(var(--muted-foreground))', fontSize: 12 }} 
                />
              )}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </motion.div>

      <motion.div variants={item} className="mt-8 pt-6 border-t">
         <h3 className="font-semibold text-lg mb-4">Model Evaluation</h3>
         <p className="text-sm text-muted-foreground mb-4">XGBoost Error Metrics</p>
         <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-secondary/30 p-4 rounded-lg hover:bg-secondary/50 transition-colors">
              <div className="text-sm text-muted-foreground mb-1">MAE</div>
              <div className="text-2xl font-mono text-indigo-400">{(data.mae).toFixed(2)}</div>
            </div>
            <div className="bg-secondary/30 p-4 rounded-lg hover:bg-secondary/50 transition-colors">
              <div className="text-sm text-muted-foreground mb-1">RMSE</div>
              <div className="text-2xl font-mono text-indigo-400">{(data.rmse).toFixed(2)}</div>
            </div>
            <div className="bg-secondary/30 p-4 rounded-lg hover:bg-secondary/50 transition-colors">
              <div className="text-sm text-muted-foreground mb-1">R² Score</div>
              <div className="text-2xl font-mono text-emerald-400">{(data.r2).toFixed(3)}</div>
            </div>
            <div className="bg-secondary/30 p-4 rounded-lg hover:bg-secondary/50 transition-colors">
              <div className="text-sm text-muted-foreground mb-1">MAPE</div>
              <div className="text-2xl font-mono text-indigo-400">{data.mape ? `${(data.mape).toFixed(2)}%` : 'N/A'}</div>
            </div>
         </div>
      </motion.div>
    </motion.div>
  );
}
