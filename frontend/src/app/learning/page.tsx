"use client";

import { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Brain, Database, Layers, Sparkles, CheckCircle2 } from "lucide-react";
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid } from "recharts";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

export default function ModelLearningPage() {
  const [data, setData] = useState<any>(null);
  const [evolution, setEvolution] = useState<any>(null);
  const [predictive, setPredictive] = useState<any>(null);
  const [selectedCat, setSelectedCat] = useState<any>(null);
  const [activeTab, setActiveTab] = useState<"contributions" | "evolution" | "graph" | "predictive" | "products">("contributions");
  const [prodSearch, setProdSearch] = useState("");
  const [prodList, setProdList] = useState<any[]>([]);
  const [prodTotal, setProdTotal] = useState(0);
  const [selectedProd, setSelectedProd] = useState<any>(null);
  const [expandedPattern, setExpandedPattern] = useState<string | null>(null);
  const [expandedAnomaly, setExpandedAnomaly] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API_BASE}/api/collective/overview`).then((r) => {
        if (!r.ok) throw new Error("Failed to load collective learning report");
        return r.json();
      }),
      fetch(`${API_BASE}/api/collective/evolution?v1=v1.0&v2=v2.0`).then((r) =>
        r.ok ? r.json() : null
      ),
      fetch(`${API_BASE}/api/collective/predictive`).then((r) =>
        r.ok ? r.json() : null
      ),
    ])
      .then(([overviewJson, evoJson, predJson]) => {
        setData(overviewJson);
        setEvolution(evoJson);
        setPredictive(predJson);
        if (overviewJson?.categories?.length > 0) {
          setSelectedCat(overviewJson.categories[0]);
        }
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message);
        setLoading(false);
      });
  }, []);

  useEffect(() => {
    if (activeTab !== "products") return;
    const params = new URLSearchParams({ limit: "25", offset: "0", sort_by: "occurrences" });
    if (prodSearch.trim()) params.set("search", prodSearch.trim());
    fetch(`${API_BASE}/api/collective/products?${params.toString()}`)
      .then((r) => (r.ok ? r.json() : null))
      .then((res) => {
        if (res) {
          setProdList(res.products || []);
          setProdTotal(res.total || 0);
          if (res.products?.length > 0 && !selectedProd) setSelectedProd(res.products[0]);
        }
      });
  }, [activeTab, prodSearch]);

  if (loading) {
    return (
      <div className="p-8 text-sm text-muted-foreground">
        Loading Collective Intelligence & Model Learning Report...
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="p-8 text-sm text-red-400">
        Error loading collective intelligence: {error}
      </div>
    );
  }

  const { overview, categories, top_products, discoveries, registry } = data;

  return (
    <div className="p-6 md:p-8 space-y-6 max-w-7xl mx-auto">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-500/10 border border-indigo-500/20 text-indigo-400 text-xs font-medium mb-2">
            <Brain className="h-3.5 w-3.5" />
            Collective Intelligence Layer
          </div>
          <h1 className="text-2xl md:text-3xl font-bold tracking-tight">
            Model Learning & Cross-Dataset Intelligence
          </h1>
          <p className="text-sm text-muted-foreground mt-1">
            Unified entity resolution, category coverage, and evidence-backed discoveries across all trained datasets.
          </p>
        </div>
        <Badge variant="outline" className="w-fit text-emerald-400 border-emerald-500/30 bg-emerald-500/10">
          Generated: {overview.generated_at}
        </Badge>
      </div>

      {/* Overview KPIs */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card className="bg-card/60 border-border/50">
          <CardContent className="pt-5">
            <div className="text-xs text-muted-foreground">Datasets Processed</div>
            <div className="text-2xl font-bold font-mono mt-1">{overview.datasets_processed}</div>
            <div className="text-[11px] text-emerald-400 mt-1">Unique deduplicated corpora</div>
          </CardContent>
        </Card>
        <Card className="bg-card/60 border-border/50">
          <CardContent className="pt-5">
            <div className="text-xs text-muted-foreground">Total Records Ingested</div>
            <div className="text-2xl font-bold font-mono mt-1">
              {overview.total_records_ingested.toLocaleString()}
            </div>
            <div className="text-[11px] text-indigo-400 mt-1">Cleaned & normalized</div>
          </CardContent>
        </Card>
        <Card className="bg-card/60 border-border/50">
          <CardContent className="pt-5">
            <div className="text-xs text-muted-foreground">Canonical Categories</div>
            <div className="text-2xl font-bold font-mono mt-1">{overview.canonical_categories_count}</div>
            <div className="text-[11px] text-muted-foreground mt-1">Cross-schema taxonomy</div>
          </CardContent>
        </Card>
        <Card className="bg-card/60 border-border/50">
          <CardContent className="pt-5">
            <div className="text-xs text-muted-foreground">Canonical Products Tracked</div>
            <div className="text-2xl font-bold font-mono mt-1">{overview.canonical_products_tracked}</div>
            <div className="text-[11px] text-muted-foreground mt-1">Top cross-dataset items</div>
          </CardContent>
        </Card>
      </div>

      {/* What the Model Learned */}
      <Card className="bg-card/60 border-border/50">
        <CardHeader>
          <CardTitle className="text-base flex items-center gap-2">
            <Sparkles className="h-4 w-4 text-indigo-400" />
            What the Model Discovered
          </CardTitle>
          <CardDescription>Evidence-backed insights with dataset provenance</CardDescription>
        </CardHeader>
        <CardContent className="grid gap-3 md:grid-cols-3">
          {discoveries.map((d: any, idx: number) => (
            <div key={idx} className="p-4 rounded-xl bg-secondary/20 border border-border/50 space-y-2">
              <div className="flex items-center justify-between gap-2">
                <span className="text-xs font-semibold text-white">{d.title}</span>
                <Badge variant="outline" className="text-[10px] text-emerald-400 border-emerald-500/30">
                  {d.confidence}
                </Badge>
              </div>
              <p className="text-xs text-zinc-300 leading-relaxed">{d.statement}</p>
              <div className="text-[10px] text-muted-foreground pt-1">
                Method: {d.method}
              </div>
            </div>
          ))}
        </CardContent>
      </Card>

      {/* Explorer Tabs */}
      <div className="flex flex-wrap gap-2 border-b border-border/50 pb-3">
        {(
          [
            ["contributions", "1. Contribution Explorer"],
            ["evolution", "2. Knowledge Evolution (v1.0 → v2.0)"],
            ["graph", "3. Cross-Dataset Relationship Graph"],
            ["predictive", "4. Predictive & Anomaly Intelligence"],
            ["products", "5. Complete Product Catalogue"],
          ] as const
        ).map(([key, label]) => (
          <button
            key={key}
            onClick={() => setActiveTab(key)}
            className={`px-4 py-2 rounded-lg text-xs font-medium transition-colors ${
              activeTab === key
                ? "bg-indigo-500 text-white"
                : "bg-secondary/40 text-muted-foreground hover:text-white"
            }`}
          >
            {label}
          </button>
        ))}
      </div>

      {activeTab === "contributions" && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <Card className="bg-card/60 border-border/50">
            <CardHeader>
              <CardTitle className="text-base flex items-center gap-2">
                <Layers className="h-4 w-4 text-indigo-400" />
                Click a Category to Inspect Dataset Contributions
              </CardTitle>
              <CardDescription>Distinguishing record frequency from dataset coverage</CardDescription>
            </CardHeader>
            <CardContent className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="border-b border-border/50 text-muted-foreground">
                    <th className="py-2">Canonical Category</th>
                    <th className="py-2">Occurrences</th>
                    <th className="py-2">Coverage</th>
                    <th className="py-2">Status</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border/30 font-mono text-[11px]">
                  {categories.map((c: any) => (
                    <tr
                      key={c.canonical_category}
                      onClick={() => setSelectedCat(c)}
                      className={`cursor-pointer transition-colors hover:bg-indigo-500/10 ${
                        selectedCat?.canonical_category === c.canonical_category ? "bg-indigo-500/15" : ""
                      }`}
                    >
                      <td className="py-2 font-sans font-medium text-white">{c.canonical_category}</td>
                      <td className="py-2">{c.total_occurrences.toLocaleString()}</td>
                      <td className="py-2 text-indigo-400">
                        {c.dataset_coverage}/{c.total_datasets} ({c.coverage_pct}%)
                      </td>
                      <td className="py-2">
                        <Badge variant="outline" className="text-[10px]">
                          {c.status}
                        </Badge>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </CardContent>
          </Card>

          <Card className="bg-card/60 border-border/50">
            <CardHeader>
              <CardTitle className="text-base">
                Contribution Breakdown: {selectedCat?.canonical_category || "Select a Category"}
              </CardTitle>
              <CardDescription>
                Total Occurrences: {selectedCat?.total_occurrences?.toLocaleString() || 0} · Coverage:{" "}
                {selectedCat?.dataset_coverage}/{selectedCat?.total_datasets} datasets
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              {(selectedCat?.contributions || []).map((cb: any) => (
                <div key={cb.dataset_id} className="space-y-1">
                  <div className="flex justify-between text-xs">
                    <span className="font-medium text-white">{cb.dataset_id}</span>
                    <span className="font-mono text-zinc-300">
                      {cb.occurrences.toLocaleString()} ({cb.contribution_pct}%)
                    </span>
                  </div>
                  <div className="w-full h-2 rounded-full bg-secondary/40 overflow-hidden">
                    <div
                      className="h-full bg-indigo-500 rounded-full"
                      style={{ width: `${Math.min(100, cb.contribution_pct)}%` }}
                    />
                  </div>
                </div>
              ))}
            </CardContent>
          </Card>
        </div>
      )}

      {activeTab === "evolution" && evolution && (
        <Card className="bg-card/60 border-border/50">
          <CardHeader>
            <CardTitle className="text-base">
              Knowledge Evolution: {evolution.v1} (4 Datasets) → {evolution.v2} (8 Unique Datasets)
            </CardTitle>
            <CardDescription>
              Tracks how category occurrences and dataset coverage expanded as additional datasets were incorporated
            </CardDescription>
          </CardHeader>
          <CardContent className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="border-b border-border/50 text-muted-foreground">
                  <th className="py-2">Canonical Category</th>
                  <th className="py-2">v1.0 Occurrences</th>
                  <th className="py-2">v2.0 Occurrences</th>
                  <th className="py-2">Delta</th>
                  <th className="py-2">Coverage Expansion</th>
                  <th className="py-2">Evolution</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border/30 font-mono text-[11px]">
                {evolution.category_evolution.map((row: any) => (
                  <tr key={row.category}>
                    <td className="py-2 font-sans font-medium text-white">{row.category}</td>
                    <td className="py-2">{row.v1_occurrences.toLocaleString()}</td>
                    <td className="py-2 text-emerald-400">{row.v2_occurrences.toLocaleString()}</td>
                    <td className="py-2 text-indigo-400">+{row.delta_occurrences.toLocaleString()}</td>
                    <td className="py-2">
                      {row.v1_coverage} → {row.v2_coverage} datasets
                    </td>
                    <td className="py-2">
                      <Badge variant="outline" className="text-[10px] text-emerald-400 border-emerald-500/30">
                        {row.status}
                      </Badge>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </CardContent>
        </Card>
      )}

      {activeTab === "predictive" && predictive && (
        <div className="space-y-6">
          {/* Summary KPIs */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <Card className="bg-card/60 border-border/50">
              <CardContent className="pt-4 pb-3">
                <div className="text-xs text-muted-foreground">Emerging Patterns</div>
                <div className="text-xl font-bold font-mono mt-1">{predictive.emerging_patterns?.length || 0}</div>
              </CardContent>
            </Card>
            <Card className="bg-card/60 border-border/50">
              <CardContent className="pt-4 pb-3">
                <div className="text-xs text-muted-foreground">Statistical Anomalies</div>
                <div className="text-xl font-bold font-mono mt-1">{predictive.anomalies?.length || 0}</div>
              </CardContent>
            </Card>
            <Card className="bg-card/60 border-border/50">
              <CardContent className="pt-4 pb-3">
                <div className="text-xs text-muted-foreground">Highest 3M Growth</div>
                <div className="text-xl font-bold font-mono text-emerald-400 mt-1">
                  +{predictive.emerging_patterns?.[0]?.growth_rate_pct ?? 0}%
                </div>
              </CardContent>
            </Card>
            <Card className="bg-card/60 border-border/50">
              <CardContent className="pt-4 pb-3">
                <div className="text-xs text-muted-foreground">Detection Threshold</div>
                <div className="text-xl font-bold font-mono text-amber-400 mt-1">|z| ≥ 2.5</div>
              </CardContent>
            </Card>
          </div>

          {/* Historical Baseline vs Recent 3M Comparison Chart */}
          {predictive.emerging_patterns?.length > 0 && (
            <Card className="bg-card/60 border-border/50">
              <CardHeader>
                <CardTitle className="text-base">Historical Baseline vs Recent 3-Month Average</CardTitle>
                <CardDescription>Monthly unit volume comparison across canonical categories</CardDescription>
              </CardHeader>
              <CardContent className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={predictive.emerging_patterns} margin={{ left: 10, right: 20 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                    <XAxis dataKey="category" tick={{ fontSize: 10, fill: "#a1a1aa" }} />
                    <YAxis tick={{ fontSize: 10, fill: "#a1a1aa" }} />
                    <Tooltip contentStyle={{ backgroundColor: "#18181b", borderColor: "#27272a", fontSize: "12px" }} />
                    <Bar dataKey="historical_baseline_monthly" name="Historical Baseline" fill="#52525b" radius={[4, 4, 0, 0]} />
                    <Bar dataKey="recent_3m_avg" name="Recent 3M Avg" fill="#6366f1" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          )}

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card className="bg-card/60 border-border/50">
              <CardHeader>
                <CardTitle className="text-base">Emerging Patterns (Recent 3M vs Baseline)</CardTitle>
                <CardDescription>Click any pattern to inspect supporting evidence</CardDescription>
              </CardHeader>
              <CardContent className="space-y-2.5">
                {(predictive.emerging_patterns || []).map((ep: any) => {
                  const isOpen = expandedPattern === ep.category;
                  const isUp = ep.growth_rate_pct > 5;
                  const isDown = ep.growth_rate_pct < -5;
                  return (
                    <div
                      key={ep.category}
                      onClick={() => setExpandedPattern(isOpen ? null : ep.category)}
                      className="p-3 rounded-lg bg-secondary/20 border border-border/40 cursor-pointer hover:bg-secondary/30 transition-colors text-xs space-y-1.5"
                    >
                      <div className="flex items-center justify-between gap-2">
                        <div>
                          <div className="font-semibold text-white">{ep.pattern_name}</div>
                          <div className="text-[11px] text-muted-foreground">
                            Baseline: {ep.historical_baseline_monthly.toLocaleString()}/mo → Recent:{" "}
                            {ep.recent_3m_avg.toLocaleString()}/mo · {ep.supporting_datasets?.length || 0} datasets
                          </div>
                        </div>
                        <Badge
                          variant="outline"
                          className={`font-mono ${
                            isUp
                              ? "text-emerald-400 border-emerald-500/30"
                              : isDown
                              ? "text-rose-400 border-rose-500/30"
                              : "text-zinc-300 border-zinc-500/30"
                          }`}
                        >
                          {ep.growth_rate_pct >= 0 ? "+" : ""}
                          {ep.growth_rate_pct}%
                        </Badge>
                      </div>
                      {isOpen && (
                        <div className="pt-2 mt-2 border-t border-border/40 text-[11px] text-zinc-300 space-y-1">
                          <div>Status: <span className="text-white">{ep.status}</span> · Confidence: <span className="text-white">{ep.confidence}</span></div>
                          <div>First Detected: <span className="font-mono text-white">{ep.first_detected}</span></div>
                          <div>Contributing Datasets: <span className="font-mono text-indigo-300">{(ep.supporting_datasets || []).join(", ")}</span></div>
                        </div>
                      )}
                    </div>
                  );
                })}
              </CardContent>
            </Card>

            <Card className="bg-card/60 border-border/50">
              <CardHeader>
                <CardTitle className="text-base">Statistical Anomaly Detection (MAD Robust Z-Score)</CardTitle>
                <CardDescription>Threshold: |z| ≥ 2.5 (excludes partial months; statistical ≠ causal demand)</CardDescription>
              </CardHeader>
              <CardContent className="space-y-2.5 max-h-96 overflow-y-auto">
                {(predictive.anomalies || []).map((an: any, i: number) => {
                  const isOpen = expandedAnomaly === i;
                  const isSpike = an.direction === "Spike";
                  return (
                    <div
                      key={i}
                      onClick={() => setExpandedAnomaly(isOpen ? null : i)}
                      className="p-3 rounded-lg bg-secondary/20 border border-border/40 cursor-pointer hover:bg-secondary/30 transition-colors text-xs space-y-1"
                    >
                      <div className="flex justify-between items-center gap-2">
                        <span className="font-semibold text-white">{an.category} · {an.period}</span>
                        <Badge
                          variant="outline"
                          className={`font-mono ${
                            isSpike ? "text-amber-400 border-amber-500/30" : "text-sky-400 border-sky-500/30"
                          }`}
                        >
                          z = {an.robust_z_score} ({an.direction})
                        </Badge>
                      </div>
                      <div className="text-[11px] text-zinc-300">{an.explanation}</div>
                      {isOpen && (
                        <div className="pt-2 mt-2 border-t border-border/40 text-[11px] text-zinc-400 space-y-1">
                          <div>Observed: <span className="font-mono text-white">{an.observed_quantity?.toLocaleString()}</span> · Median: <span className="font-mono text-white">{an.expected_median?.toLocaleString()}</span></div>
                          <div>Note: Statistical deviation detected via Median/MAD; verify seasonal or bulk-order context before assuming organic demand shift.</div>
                        </div>
                      )}
                    </div>
                  );
                })}
              </CardContent>
            </Card>
          </div>
        </div>
      )}

      {activeTab === "products" && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <Card className="bg-card/60 border-border/50">
            <CardHeader>
              <CardTitle className="text-base">
                Complete Product Catalogue ({prodTotal.toLocaleString()} matching)
              </CardTitle>
              <CardDescription>Server-side search & pagination across all 8 unique datasets</CardDescription>
              <input
                type="text"
                value={prodSearch}
                onChange={(e) => setProdSearch(e.target.value)}
                placeholder="Search canonical product (e.g. Bag, Mug, Heart)..."
                className="mt-2 w-full bg-background/80 border border-border/60 rounded-md px-3 py-1.5 text-xs text-foreground focus:outline-none focus:ring-1 focus:ring-indigo-500"
              />
            </CardHeader>
            <CardContent className="overflow-x-auto max-h-96 overflow-y-auto">
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="border-b border-border/50 text-muted-foreground">
                    <th className="py-2">Canonical Product</th>
                    <th className="py-2">Category</th>
                    <th className="py-2">Occurrences</th>
                    <th className="py-2">Coverage</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border/30 font-mono text-[11px]">
                  {prodList.map((p: any) => (
                    <tr
                      key={p.canonical_product}
                      onClick={() => setSelectedProd(p)}
                      className={`cursor-pointer hover:bg-indigo-500/10 ${
                        selectedProd?.canonical_product === p.canonical_product ? "bg-indigo-500/15" : ""
                      }`}
                    >
                      <td className="py-2 font-sans font-medium text-white">{p.canonical_product}</td>
                      <td className="py-2 font-sans text-zinc-400">{p.canonical_category}</td>
                      <td className="py-2">{p.total_occurrences.toLocaleString()}</td>
                      <td className="py-2 text-indigo-400">
                        {p.dataset_coverage}/{p.total_datasets}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </CardContent>
          </Card>

          <Card className="bg-card/60 border-border/50">
            <CardHeader>
              <CardTitle className="text-base">
                Product Provenance: {selectedProd?.canonical_product || "Select a Product"}
              </CardTitle>
              <CardDescription>
                Category: {selectedProd?.canonical_category} · Original names:{" "}
                {(selectedProd?.original_names || []).slice(0, 3).join(", ")}
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              {(selectedProd?.contributions || []).map((cb: any) => (
                <div key={cb.dataset_id} className="space-y-1">
                  <div className="flex justify-between text-xs">
                    <span className="font-medium text-white">{cb.dataset_id}</span>
                    <span className="font-mono text-zinc-300">
                      {cb.occurrences.toLocaleString()} ({cb.contribution_pct}%)
                    </span>
                  </div>
                  <div className="w-full h-2 rounded-full bg-secondary/40 overflow-hidden">
                    <div
                      className="h-full bg-emerald-500 rounded-full"
                      style={{ width: `${Math.min(100, cb.contribution_pct)}%` }}
                    />
                  </div>
                </div>
              ))}
            </CardContent>
          </Card>
        </div>
      )}

      {activeTab === "graph" && selectedCat && (
        <Card className="bg-card/60 border-border/50">
          <CardHeader>
            <CardTitle className="text-base">
              Cross-Dataset Relationship Graph: {selectedCat.canonical_category}
            </CardTitle>
            <CardDescription>
              Visualizing dataset-to-category contribution edges weighted by observed record share
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex flex-col md:flex-row items-center justify-around gap-6 p-6 rounded-xl bg-secondary/20 border border-border/40">
              <div className="space-y-2 w-full md:w-1/2">
                <div className="text-xs font-semibold text-muted-foreground uppercase mb-2">
                  Contributing Dataset Nodes
                </div>
                {(selectedCat.contributions || []).map((cb: any) => (
                  <div
                    key={cb.dataset_id}
                    className="flex items-center justify-between p-2.5 rounded-lg bg-background/80 border border-border/50 text-xs"
                  >
                    <span className="font-mono text-indigo-400">{cb.dataset_id}</span>
                    <span className="text-zinc-400">── {cb.contribution_pct}% ──▶</span>
                  </div>
                ))}
              </div>
              <div className="p-6 rounded-2xl bg-indigo-500/15 border border-indigo-500/40 text-center space-y-1">
                <div className="text-[10px] uppercase tracking-wider text-indigo-300">Canonical Category</div>
                <div className="text-lg font-bold text-white">{selectedCat.canonical_category}</div>
                <div className="text-xs font-mono text-emerald-400">
                  {selectedCat.total_occurrences.toLocaleString()} records · {selectedCat.dataset_coverage}/
                  {selectedCat.total_datasets} datasets
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {activeTab === "predictive" && predictive && (
        <div className="space-y-6">
          {/* Summary KPIs */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <Card className="bg-card/60 border-border/50">
              <CardContent className="pt-4 pb-3">
                <div className="text-xs text-muted-foreground">Emerging Patterns</div>
                <div className="text-xl font-bold font-mono mt-1">{predictive.emerging_patterns?.length || 0}</div>
              </CardContent>
            </Card>
            <Card className="bg-card/60 border-border/50">
              <CardContent className="pt-4 pb-3">
                <div className="text-xs text-muted-foreground">Statistical Anomalies</div>
                <div className="text-xl font-bold font-mono mt-1">{predictive.anomalies?.length || 0}</div>
              </CardContent>
            </Card>
            <Card className="bg-card/60 border-border/50">
              <CardContent className="pt-4 pb-3">
                <div className="text-xs text-muted-foreground">Highest 3M Growth</div>
                <div className="text-xl font-bold font-mono text-emerald-400 mt-1">
                  +{predictive.emerging_patterns?.[0]?.growth_rate_pct ?? 0}%
                </div>
              </CardContent>
            </Card>
            <Card className="bg-card/60 border-border/50">
              <CardContent className="pt-4 pb-3">
                <div className="text-xs text-muted-foreground">Detection Threshold</div>
                <div className="text-xl font-bold font-mono text-amber-400 mt-1">|z| ≥ 2.5</div>
              </CardContent>
            </Card>
          </div>

          {/* Historical Baseline vs Recent 3M Comparison Chart */}
          {predictive.emerging_patterns?.length > 0 && (
            <Card className="bg-card/60 border-border/50">
              <CardHeader>
                <CardTitle className="text-base">Historical Baseline vs Recent 3-Month Average</CardTitle>
                <CardDescription>Monthly unit volume comparison across canonical categories</CardDescription>
              </CardHeader>
              <CardContent className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={predictive.emerging_patterns} margin={{ left: 10, right: 20 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                    <XAxis dataKey="category" tick={{ fontSize: 10, fill: "#a1a1aa" }} />
                    <YAxis tick={{ fontSize: 10, fill: "#a1a1aa" }} />
                    <Tooltip contentStyle={{ backgroundColor: "#18181b", borderColor: "#27272a", fontSize: "12px" }} />
                    <Bar dataKey="historical_baseline_monthly" name="Historical Baseline" fill="#52525b" radius={[4, 4, 0, 0]} />
                    <Bar dataKey="recent_3m_avg" name="Recent 3M Avg" fill="#6366f1" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          )}

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card className="bg-card/60 border-border/50">
              <CardHeader>
                <CardTitle className="text-base">Emerging Patterns (Recent 3M vs Baseline)</CardTitle>
                <CardDescription>Click any pattern to inspect supporting evidence</CardDescription>
              </CardHeader>
              <CardContent className="space-y-2.5">
                {(predictive.emerging_patterns || []).map((ep: any) => {
                  const isOpen = expandedPattern === ep.category;
                  const isUp = ep.growth_rate_pct > 5;
                  const isDown = ep.growth_rate_pct < -5;
                  return (
                    <div
                      key={ep.category}
                      onClick={() => setExpandedPattern(isOpen ? null : ep.category)}
                      className="p-3 rounded-lg bg-secondary/20 border border-border/40 cursor-pointer hover:bg-secondary/30 transition-colors text-xs space-y-1.5"
                    >
                      <div className="flex items-center justify-between gap-2">
                        <div>
                          <div className="font-semibold text-white">{ep.pattern_name}</div>
                          <div className="text-[11px] text-muted-foreground">
                            Baseline: {ep.historical_baseline_monthly.toLocaleString()}/mo → Recent:{" "}
                            {ep.recent_3m_avg.toLocaleString()}/mo · {ep.supporting_datasets?.length || 0} datasets
                          </div>
                        </div>
                        <Badge
                          variant="outline"
                          className={`font-mono ${
                            isUp
                              ? "text-emerald-400 border-emerald-500/30"
                              : isDown
                              ? "text-rose-400 border-rose-500/30"
                              : "text-zinc-300 border-zinc-500/30"
                          }`}
                        >
                          {ep.growth_rate_pct >= 0 ? "+" : ""}
                          {ep.growth_rate_pct}%
                        </Badge>
                      </div>
                      {isOpen && (
                        <div className="pt-2 mt-2 border-t border-border/40 text-[11px] text-zinc-300 space-y-1">
                          <div>Status: <span className="text-white">{ep.status}</span> · Confidence: <span className="text-white">{ep.confidence}</span></div>
                          <div>First Detected: <span className="font-mono text-white">{ep.first_detected}</span></div>
                          <div>Contributing Datasets: <span className="font-mono text-indigo-300">{(ep.supporting_datasets || []).join(", ")}</span></div>
                        </div>
                      )}
                    </div>
                  );
                })}
              </CardContent>
            </Card>

            <Card className="bg-card/60 border-border/50">
              <CardHeader>
                <CardTitle className="text-base">Statistical Anomaly Detection (MAD Robust Z-Score)</CardTitle>
                <CardDescription>Threshold: |z| ≥ 2.5 (excludes partial months; statistical ≠ causal demand)</CardDescription>
              </CardHeader>
              <CardContent className="space-y-2.5 max-h-96 overflow-y-auto">
                {(predictive.anomalies || []).map((an: any, i: number) => {
                  const isOpen = expandedAnomaly === i;
                  const isSpike = an.direction === "Spike";
                  return (
                    <div
                      key={i}
                      onClick={() => setExpandedAnomaly(isOpen ? null : i)}
                      className="p-3 rounded-lg bg-secondary/20 border border-border/40 cursor-pointer hover:bg-secondary/30 transition-colors text-xs space-y-1"
                    >
                      <div className="flex justify-between items-center gap-2">
                        <span className="font-semibold text-white">{an.category} · {an.period}</span>
                        <Badge
                          variant="outline"
                          className={`font-mono ${
                            isSpike ? "text-amber-400 border-amber-500/30" : "text-sky-400 border-sky-500/30"
                          }`}
                        >
                          z = {an.robust_z_score} ({an.direction})
                        </Badge>
                      </div>
                      <div className="text-[11px] text-zinc-300">{an.explanation}</div>
                      {isOpen && (
                        <div className="pt-2 mt-2 border-t border-border/40 text-[11px] text-zinc-400 space-y-1">
                          <div>Observed: <span className="font-mono text-white">{an.observed_quantity?.toLocaleString()}</span> · Median: <span className="font-mono text-white">{an.expected_median?.toLocaleString()}</span></div>
                          <div>Note: Statistical deviation detected via Median/MAD; verify seasonal or bulk-order context before assuming organic demand shift.</div>
                        </div>
                      )}
                    </div>
                  );
                })}
              </CardContent>
            </Card>
          </div>
        </div>
      )}

      {activeTab === "products" && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <Card className="bg-card/60 border-border/50">
            <CardHeader>
              <CardTitle className="text-base">
                Complete Product Catalogue ({prodTotal.toLocaleString()} matching)
              </CardTitle>
              <CardDescription>Server-side search & pagination across all 8 unique datasets</CardDescription>
              <input
                type="text"
                value={prodSearch}
                onChange={(e) => setProdSearch(e.target.value)}
                placeholder="Search canonical product (e.g. Bag, Mug, Heart)..."
                className="mt-2 w-full bg-background/80 border border-border/60 rounded-md px-3 py-1.5 text-xs text-foreground focus:outline-none focus:ring-1 focus:ring-indigo-500"
              />
            </CardHeader>
            <CardContent className="overflow-x-auto max-h-96 overflow-y-auto">
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="border-b border-border/50 text-muted-foreground">
                    <th className="py-2">Canonical Product</th>
                    <th className="py-2">Category</th>
                    <th className="py-2">Occurrences</th>
                    <th className="py-2">Coverage</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border/30 font-mono text-[11px]">
                  {prodList.map((p: any) => (
                    <tr
                      key={p.canonical_product}
                      onClick={() => setSelectedProd(p)}
                      className={`cursor-pointer hover:bg-indigo-500/10 ${
                        selectedProd?.canonical_product === p.canonical_product ? "bg-indigo-500/15" : ""
                      }`}
                    >
                      <td className="py-2 font-sans font-medium text-white">{p.canonical_product}</td>
                      <td className="py-2 font-sans text-zinc-400">{p.canonical_category}</td>
                      <td className="py-2">{p.total_occurrences.toLocaleString()}</td>
                      <td className="py-2 text-indigo-400">
                        {p.dataset_coverage}/{p.total_datasets}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </CardContent>
          </Card>

          <Card className="bg-card/60 border-border/50">
            <CardHeader>
              <CardTitle className="text-base">
                Product Provenance: {selectedProd?.canonical_product || "Select a Product"}
              </CardTitle>
              <CardDescription>
                Category: {selectedProd?.canonical_category} · Original names:{" "}
                {(selectedProd?.original_names || []).slice(0, 3).join(", ")}
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              {(selectedProd?.contributions || []).map((cb: any) => (
                <div key={cb.dataset_id} className="space-y-1">
                  <div className="flex justify-between text-xs">
                    <span className="font-medium text-white">{cb.dataset_id}</span>
                    <span className="font-mono text-zinc-300">
                      {cb.occurrences.toLocaleString()} ({cb.contribution_pct}%)
                    </span>
                  </div>
                  <div className="w-full h-2 rounded-full bg-secondary/40 overflow-hidden">
                    <div
                      className="h-full bg-emerald-500 rounded-full"
                      style={{ width: `${Math.min(100, cb.contribution_pct)}%` }}
                    />
                  </div>
                </div>
              ))}
            </CardContent>
          </Card>
        </div>
      )}

      {/* Dataset Registry */}
      <Card className="bg-card/60 border-border/50">
        <CardHeader>
          <CardTitle className="text-base flex items-center gap-2">
            <Database className="h-4 w-4 text-emerald-400" />
            Centralized Dataset Registry
          </CardTitle>
          <CardDescription>Provenance and schema audit for every ingested dataset</CardDescription>
        </CardHeader>
        <CardContent className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead>
              <tr className="border-b border-border/50 text-muted-foreground">
                <th className="py-2">Dataset</th>
                <th className="py-2">Records</th>
                <th className="py-2">Transactions</th>
                <th className="py-2">Customers</th>
                <th className="py-2">Date Range</th>
                <th className="py-2">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border/30 font-mono text-[11px]">
              {registry.map((r: any) => (
                <tr key={r.dataset_id}>
                  <td className="py-2 font-sans font-medium text-white">{r.dataset_name}</td>
                  <td className="py-2">{r.records.toLocaleString()}</td>
                  <td className="py-2">{r.unique_transactions.toLocaleString()}</td>
                  <td className="py-2">{r.unique_customers.toLocaleString()}</td>
                  <td className="py-2 text-zinc-400">{r.date_range}</td>
                  <td className="py-2">
                    <span className="inline-flex items-center gap-1 text-emerald-400">
                      <CheckCircle2 className="h-3 w-3" /> {r.status}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </CardContent>
      </Card>
    </div>
  );
}
