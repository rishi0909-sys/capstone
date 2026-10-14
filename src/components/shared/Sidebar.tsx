"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Activity, Database, LineChart, PieChart, Sparkles, TrendingUp, Network } from "lucide-react";

export function Sidebar() {
  const pathname = usePathname();

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
      </nav>
      <div className="p-4 border-t border-border/50 text-xs text-muted-foreground bg-secondary/20">
        Knowledge Base v1.0<br/>Precomputed Intelligence
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
