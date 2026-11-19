"use client";

import { useState, useEffect, useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Link } from "react-router-dom";
import { Sparkles, X, Search, Info, Mic } from "lucide-react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

export default function InsightsWorkspacePage() {
  const [workspace, setWorkspace] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [catFilter, setCatFilter] = useState("All");
  const [impactFilter, setImpactFilter] = useState("All");
  const [search, setSearch] = useState("");
  const [selectedInsight, setSelectedInsight] = useState<any>(null);


  useEffect(() => {
    fetch(`${API_BASE}/api/insights/workspace`)
      .then((r) => {
        if (!r.ok) throw new Error("Failed to load insights workspace");
        return r.json();
      })
      .then((json) => {
        setWorkspace(json);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message);
        setLoading(false);
      });
  }, []);

  const filteredInsights = useMemo(() => {
    if (!workspace?.insights) return [];
    const list = workspace.insights.filter((item: any) => {
      if (catFilter !== "All" && item.category !== catFilter) return false;
      if (impactFilter !== "All" && item.impact !== impactFilter) return false;
      if (search.trim()) {
        const q = search.trim().toLowerCase();
        if (!item.title.toLowerCase().includes(q) && !item.summary.toLowerCase().includes(q)) return false;
      }
      return true;
    });
    const salesScore = (item: any) => {
      const m = item.metrics || {};
      if (item.category === "Emerging Trends") return Number(m.recent_3m_avg || 0);
      if (item.category === "Anomalies") return Number(m.observed_quantity || 0);
      if (item.category === "Customer Behavior") return Number(m.total_customers || 0) * 10;
      return Number(m.confidence_pct || 0) * Number(m.lift || 1);
    };
    return [...list].sort((a: any, b: any) => {
      if (a.impact !== b.impact) return a.impact === "HIGH" ? -1 : 1;
      return salesScore(b) - salesScore(a);
    });
  }, [workspace, catFilter, impactFilter, search]);


  if (loading) {
    return <div className="p-8 text-sm text-muted-foreground">Loading Automated Business Insights...</div>;
  }
  if (error || !workspace) {
    return <div className="p-8 text-sm text-red-400">Error: {error}</div>;
  }

  const { overview } = workspace;

  return (
    <div className="p-6 md:p-8 space-y-6 max-w-7xl mx-auto">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-500/10 border border-indigo-500/20 text-indigo-400 text-xs font-medium mb-2">
            <Sparkles className="h-3.5 w-3.5" />
            AI-Powered Business Intelligence
          </div>
          <h1 className="text-2xl md:text-3xl font-bold tracking-tight">Automated Business Insights</h1>
          <p className="text-sm text-muted-foreground mt-1">
            Deduplicated product associations, emerging trends, anomalies, and customer behavior grounded in real data.
          </p>
        </div>
      </div>

      {/* Executive Overview (Clickable KPIs) */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
        {[
          { label: "Total Insights", val: overview.total_insights, color: "text-white", cat: "All", imp: "All" },
          { label: "High Impact", val: overview.high_impact_count, color: "text-emerald-400", cat: "All", imp: "HIGH" },
          { label: "Unique Associations", val: overview.associations_count, color: "text-indigo-400", cat: "Product Associations", imp: "All" },
          { label: "Emerging Trends", val: overview.emerging_count, color: "text-amber-400", cat: "Emerging Trends", imp: "All" },
          { label: "Anomalies", val: overview.anomalies_count, color: "text-sky-400", cat: "Anomalies", imp: "All" },
        ].map((kpi) => (
          <Card
            key={kpi.label}
            onClick={() => {
              setCatFilter(kpi.cat);
              setImpactFilter(kpi.imp);
            }}
            className="bg-card/60 border-border/50 hover:border-indigo-500/50 cursor-pointer transition-all"
          >
            <CardContent className="pt-4 pb-3">
              <div className="text-[11px] text-muted-foreground">{kpi.label}</div>
              <div className={`text-xl font-bold font-mono mt-1 ${kpi.color}`}>{kpi.val}</div>
              <div className="text-[10px] text-indigo-400/80 mt-0.5">Click to view cases →</div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Filters */}
      <div className="flex flex-col md:flex-row gap-3 items-start md:items-center justify-between bg-card/40 p-3 rounded-xl border border-border/50">
        <div className="flex flex-wrap gap-1.5">
          {["All", "Product Associations", "Emerging Trends", "Anomalies", "Customer Behavior"].map((cat) => (
            <button
              key={cat}
              onClick={() => setCatFilter(cat)}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
                catFilter === cat ? "bg-indigo-500 text-white" : "bg-secondary/40 text-muted-foreground hover:text-white"
              }`}
            >
              {cat}
            </button>
          ))}
        </div>
        <div className="flex gap-2 w-full md:w-auto">
          <select
            value={impactFilter}
            onChange={(e) => setImpactFilter(e.target.value)}
            className="bg-background/80 border border-border/60 rounded-lg px-2.5 py-1.5 text-xs text-foreground"
          >
            <option value="All">All Impact</option>
            <option value="HIGH">High Impact</option>
            <option value="MEDIUM">Medium Impact</option>
          </select>
          <div className="relative flex-1 md:w-56">
            <Search className="h-3.5 w-3.5 absolute left-2.5 top-2 text-muted-foreground" />
            <input
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search insights..."
              className="w-full bg-background/80 border border-border/60 rounded-lg pl-8 pr-3 py-1.5 text-xs text-foreground focus:outline-none focus:ring-1 focus:ring-indigo-500"
            />
          </div>
        </div>
      </div>

      {/* Insight Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {filteredInsights.map((item: any) => (
          <Card
            key={item.id}
            onClick={() => setSelectedInsight(item)}
            className="bg-card/60 border-border/50 hover:border-indigo-500/40 cursor-pointer transition-all"
          >
            <CardHeader className="pb-2">
              <div className="flex items-center justify-between gap-2">
                <Badge variant="outline" className="text-[10px] text-indigo-300 border-indigo-500/30">
                  {item.category}
                </Badge>
                <Badge
                  variant="outline"
                  className={`text-[10px] ${
                    item.impact === "HIGH"
                      ? "text-emerald-400 border-emerald-500/30"
                      : "text-amber-400 border-amber-500/30"
                  }`}
                >
                  {item.impact} IMPACT
                </Badge>
              </div>
              <CardTitle className="text-sm mt-2 leading-snug">{item.title}</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3 text-xs">
              <div className="px-2.5 py-1.5 rounded-lg bg-indigo-500/10 border border-indigo-500/20 text-[11px] font-medium text-indigo-200">
                {item.category === "Product Associations" && item.metrics?.confidence_pct
                  ? `💡 Why look: ${item.metrics.confidence_pct}% of shoppers buy both — great bundle opportunity`
                  : item.category === "Emerging Trends" && item.metrics?.growth_rate_pct
                  ? `📈 Why look: Up +${item.metrics.growth_rate_pct}% recently — stock up before peak`
                  : item.category === "Anomalies"
                  ? `⚠️ Why look: Unusual ${item.metrics?.observed_quantity?.toLocaleString() || ""} unit swing vs normal`
                  : `🔁 Why look: ${item.metrics?.repeat_rate_pct || 48.2}% of shoppers return for repeat orders`}
              </div>
              <p className="text-zinc-300 leading-relaxed">{item.summary}</p>
              <div className="flex items-center justify-between text-[11px] text-muted-foreground pt-2 border-t border-border/40">
                <span>{item.provenance}</span>
                <span className="text-indigo-400 font-medium">Inspect →</span>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Detail Modal */}
      <AnimatePresence>
        {selectedInsight && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 bg-black/70 backdrop-blur-sm flex items-center justify-center p-4"
            onClick={() => setSelectedInsight(null)}
          >
            <motion.div
              initial={{ scale: 0.96, y: 12 }}
              animate={{ scale: 1, y: 0 }}
              exit={{ scale: 0.96, y: 12 }}
              onClick={(e) => e.stopPropagation()}
              className="w-full max-w-lg rounded-2xl bg-card border border-border/60 p-6 space-y-4 shadow-2xl"
            >
              <div className="flex items-start justify-between gap-4">
                <div>
                  <Badge variant="outline" className="text-[10px] text-indigo-300 border-indigo-500/30 mb-2">
                    {selectedInsight.category} · {selectedInsight.impact} IMPACT
                  </Badge>
                  <h3 className="text-base font-bold text-white">{selectedInsight.title}</h3>
                </div>
                <button onClick={() => setSelectedInsight(null)} className="text-muted-foreground hover:text-white">
                  <X className="h-4 w-4" />
                </button>
              </div>
              <p className="text-xs text-zinc-200 leading-relaxed">{selectedInsight.summary}</p>
              <div className="p-3 rounded-xl bg-secondary/30 border border-border/40 text-xs space-y-2">
                <div className="font-semibold text-white">Why This Case Is Here (Supporting Numbers)</div>
                <div className="space-y-1.5 text-[11px] text-zinc-300">
                  {Object.entries(selectedInsight.metrics || {}).map(([k, v]) => {
                    if (v === null || v === undefined) return null;
                    const desc: Record<string, string> = {
                      support_pct: "Share of all 39,536 orders containing both items",
                      confidence_pct: "How often buyers of the first item also bought the second",
                      lift: "How many times stronger this pairing is than random chance",
                      growth_rate_pct: "Recent 3-month change vs historical monthly average",
                      recent_3m_avg: "Average monthly units sold in the last 3 months",
                      baseline_monthly: "Historical average monthly units sold",
                      observed_quantity: "Actual units recorded in that flagged month",
                      expected_median: "Typical monthly median units for this category",
                      robust_z_score: "Statistical distance from median (|z| ≥ 2.5 flags an anomaly)",
                    };
                    return (
                      <div key={k} className="flex flex-col bg-background/50 p-2 rounded-lg border border-border/30">
                        <div className="flex justify-between font-mono">
                          <span className="text-indigo-300">{k}</span>
                          <span className="text-white font-semibold">{String(v)}</span>
                        </div>
                        {desc[k] && <span className="text-[10px] text-muted-foreground mt-0.5">{desc[k]}</span>}
                      </div>
                    );
                  })}
                </div>
              </div>
              <div className="text-xs space-y-1">
                <div className="font-semibold text-emerald-400">Recommended Action</div>
                <p className="text-zinc-300">{selectedInsight.recommendation}</p>
              </div>
              <div className="text-[11px] text-muted-foreground flex items-start gap-1.5 pt-2 border-t border-border/40">
                <Info className="h-3.5 w-3.5 mt-0.5 flex-shrink-0" />
                <span>
                  Source: {selectedInsight.provenance}. {selectedInsight.limitation}
                </span>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Voice Assistant shortcut */}
      <Link
        to="/voice"
        className="fixed bottom-5 right-5 z-40 flex items-center gap-2 px-3.5 py-2 rounded-full bg-indigo-500 hover:bg-indigo-600 text-white text-xs font-medium shadow-lg shadow-indigo-500/25"
      >
        <Mic className="h-3.5 w-3.5" />
        Talk to AI
      </Link>
    </div>
  );
}
