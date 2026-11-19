"use client";

import { useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Activity, Database, LineChart, PieChart, Sparkles, TrendingUp, Network, Brain, Mic } from "lucide-react";

const TRAINED_DATASETS = [
  { id: "dataset_3", label: "Dataset 3 · UCI 2-Yr (1.04M)", badge: "Active · 20.4% WAPE" },
  { id: "dataset_1", label: "Dataset 1 · Clean UK (400K)", badge: "Exp 1 · Baseline" },
  { id: "dataset_2", label: "Dataset 2 · Raw + Guests (528K)", badge: "Exp 2 · 24.5% WAPE" },
  { id: "dataset_4", label: "Dataset 4 · E-Commerce (5K)", badge: "Exp 4 · 16.7% WAPE" },
  { id: "dataset_5", label: "Dataset 5 · E-Com Verified", badge: "Exp 5 · Deduped" },
  { id: "dataset_6", label: "Dataset 6 · Retail CSV (530K)", badge: "Exp 6 · 435 Rules" },
  { id: "dataset_7", label: "Dataset 7 · Retail XLSX (530K)", badge: "Exp 7 · 24.2% WAPE" },
  { id: "dataset_8", label: "Dataset 8 · Customer XLSX", badge: "Exp 8 · Deduped" },
  { id: "dataset_9", label: "Dataset 9 · 2009-11 CSVs (1.04M)", badge: "Exp 9 · 21.9% WAPE" },
  { id: "dataset_10", label: "Dataset 10 · Unified CSV (530K)", badge: "Exp 10 · 435 Rules" },
];

export function Sidebar() {
  const pathname = usePathname();
  const [selectedDs, setSelectedDs] = useState("dataset_3");

  // Hide sidebar on landing page
  if (pathname === "/") return null;

  return (
    <aside className="w-64 border-r border-border bg-card/80 backdrop-blur-xl flex-shrink-0 flex flex-col hidden md:flex transition-all duration-300">
      <div className="h-16 flex items-center px-6 border-b border-border/50">
        <Sparkles className="h-5 w-5 text-indigo-500 mr-2" />
        <span className="font-semibold text-lg tracking-tight">Retail Intelligence</span>
      </div>
      <nav className="flex-1 py-4 px-3 space-y-2 overflow-y-auto">
        <div className="pt-2 pb-1 px-3">
          <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">Explore</p>
        </div>
        <NavItem href="/dashboard" icon={<Activity className="h-4 w-4" />} label="Categories" active={pathname === "/dashboard"} />
        <NavItem href="/patterns" icon={<Network className="h-4 w-4" />} label="Product Patterns" active={pathname === "/patterns"} />
        <NavItem href="/demographics" icon={<PieChart className="h-4 w-4" />} label="Customer Behavior" active={pathname === "/demographics"} />
        <NavItem href="/temporal" icon={<LineChart className="h-4 w-4" />} label="Trends" active={pathname === "/temporal"} />
        <NavItem href="/forecasting" icon={<TrendingUp className="h-4 w-4" />} label="Forecast" active={pathname === "/forecasting"} />
        
        <div className="pt-4 pb-1 px-3">
          <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">Understand</p>
        </div>
        
        <NavItem href="/insights" icon={<Sparkles className="h-4 w-4" />} label="AI Explanations" active={pathname === "/insights"} />
        <NavItem href="/learning" icon={<Brain className="h-4 w-4" />} label="Model Learning" active={pathname === "/learning"} />
        <NavItem href="/voice" icon={<Mic className="h-4 w-4" />} label="Voice Assistant" active={pathname === "/voice"} />
      </nav>
      <div className="p-4 border-t border-border/50 text-xs text-muted-foreground bg-secondary/20 space-y-2">
        <div className="flex items-center justify-between text-[11px] font-semibold text-indigo-400">
          <span className="flex items-center gap-1.5">
            <Database className="h-3.5 w-3.5" />
            10 Trained Datasets
          </span>
          <span className="px-1.5 py-0.5 rounded bg-indigo-500/10 text-[10px]">Multi-Corpus</span>
        </div>
        <select
          value={selectedDs}
          onChange={(e) => setSelectedDs(e.target.value)}
          className="w-full bg-background/80 border border-border/60 rounded-md px-2 py-1.5 text-[11px] text-foreground focus:outline-none focus:ring-1 focus:ring-indigo-500"
        >
          {TRAINED_DATASETS.map((ds) => (
            <option key={ds.id} value={ds.id}>
              {ds.label}
            </option>
          ))}
        </select>
        <div className="text-[10px] text-muted-foreground/80">
          {TRAINED_DATASETS.find((d) => d.id === selectedDs)?.badge} · HGB + XGBoost
        </div>
      </div>
    </aside>
  );
}

function NavItem({ href, icon, label, active }: { href: string; icon: React.ReactNode; label: string; active: boolean }) {
  return (
    <Link 
      href={href} 
      className={`flex items-center px-3 py-2.5 text-sm font-medium rounded-lg transition-all duration-200 group ${
        active 
          ? "bg-indigo-500/10 text-indigo-400" 
          : "text-muted-foreground hover:bg-secondary/80 hover:text-foreground"
      }`}
    >
      <span className={`mr-3 transition-colors ${active ? "text-indigo-400" : "text-muted-foreground group-hover:text-foreground"}`}>
        {icon}
      </span>
      {label}
      {active && (
        <span className="ml-auto w-1 h-4 bg-indigo-500 rounded-full" />
      )}
    </Link>
  );
}
