"use client";

import { useState } from "react";
import { useAppStore } from "@/lib/store/useAppStore";
import { Activity, Search, RefreshCw, AlertTriangle, FileText } from "lucide-react";

const segments = [
  { id: 1, name: "Gen Z Enthusiasts", sub: "18-25 • urban-core", status: "degraded" },
  { id: 2, name: "Millennial Parents", sub: "26-35 • suburban", status: "healthy" },
  { id: 3, name: "Gen X Professionals", sub: "36-50 • high-income", status: "healthy" },
  { id: 4, name: "Boomer Traditionalists", sub: "51+ • rural", status: "healthy" },
  { id: 5, name: "Student Budget", sub: "18-22 • university", status: "error" },
  { id: 6, name: "Tech Early Adopters", sub: "25-40 • coastal", status: "healthy" },
];

export default function DemographicsPage() {
  const { activeCategory } = useAppStore();
  const [activeSegment, setActiveSegment] = useState(segments[0]);

  return (
    <div className="min-h-screen bg-[#0a0a0a] text-zinc-300 font-sans p-6 rounded-3xl border border-white/5 shadow-2xl">
      {/* Top Header */}
      <div className="flex items-center justify-between mb-6 border-b border-white/10 pb-4">
        <div className="flex items-center gap-4">
           <div className="flex items-center gap-2 px-3 py-1.5 bg-[#141414] border border-white/10 rounded-md">
             <Activity className="h-4 w-4 text-zinc-400" />
             <span className="text-sm font-medium">Demographic Health</span>
             <span className="bg-zinc-800 text-xs px-2 py-0.5 rounded-full ml-2">9</span>
           </div>
        </div>
        <div className="flex items-center gap-4 text-sm">
           <span className="flex items-center text-emerald-500 gap-2 font-medium">
             <span className="h-2 w-2 rounded-full bg-emerald-500 animate-pulse"></span>
             Live • 5s
           </span>
           <button className="flex items-center gap-2 bg-white/5 hover:bg-white/10 border border-white/10 px-3 py-1.5 rounded-md transition-colors text-white">
             <RefreshCw className="h-3.5 w-3.5" /> Refresh
           </button>
        </div>
      </div>

      <div className="flex flex-col lg:flex-row gap-6">
        {/* Sidebar */}
        <div className="w-full lg:w-[320px] shrink-0 bg-[#111111] border border-white/10 rounded-xl overflow-hidden flex flex-col h-[750px]">
           <div className="p-4 border-b border-white/10">
             <div className="relative">
               <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-zinc-500" />
               <input 
                 type="text" 
                 placeholder="Filter segments..." 
                 className="w-full bg-[#1a1a1a] border border-white/10 rounded-md pl-9 pr-4 py-2 text-sm focus:outline-none focus:border-zinc-500 transition-colors text-white"
               />
             </div>
           </div>
           <div className="flex-1 overflow-y-auto custom-scrollbar">
             {segments.map((seg) => (
               <div 
                 key={seg.id}
                 onClick={() => setActiveSegment(seg)}
                 className={`p-4 border-b border-white/5 flex items-center justify-between cursor-pointer transition-colors ${activeSegment.id === seg.id ? 'bg-white/5' : 'hover:bg-white/5'}`}
               >
                 <div className="flex items-start gap-3">
                   <div className={`mt-1.5 h-1.5 w-1.5 rounded-full ${seg.status === 'healthy' ? 'bg-zinc-600' : seg.status === 'degraded' ? 'bg-amber-500' : 'bg-red-500'}`} />
                   <div>
                     <div className={`text-sm font-semibold ${activeSegment.id === seg.id ? 'text-white' : 'text-zinc-300'}`}>{seg.name}</div>
                     <div className="text-[11px] text-zinc-500 mt-0.5">{seg.sub}</div>
                   </div>
                 </div>
                 {/* Mini sparkline */}
                 <svg width="40" height="15" viewBox="0 0 40 15" className="opacity-70">
                   <path d="M 0 10 Q 5 2 10 8 T 20 5 T 30 12 T 40 5" fill="none" stroke={seg.status === 'degraded' ? '#f59e0b' : seg.status === 'error' ? '#ef4444' : '#71717a'} strokeWidth="1.5" />
                 </svg>
               </div>
             ))}
           </div>
        </div>

        {/* Main Content Area */}
        <div className="flex-1 bg-[#111111] border border-white/10 rounded-xl p-8 h-[750px] overflow-y-auto custom-scrollbar">
           {/* Main Header */}
           <div className="flex justify-between items-start mb-8">
             <div>
               <h2 className="text-2xl font-semibold text-white flex items-center gap-3">
                 {activeSegment.name} 
                 {activeSegment.status === 'degraded' && <span className="text-amber-500 text-sm flex items-center gap-1.5 bg-amber-500/10 px-2 py-0.5 rounded-md"><AlertTriangle className="h-4 w-4"/> Degraded</span>}
               </h2>
               <div className="text-xs text-zinc-500 mt-2 flex items-center gap-4">
                 <span className="flex items-center gap-1.5"><FileText className="h-3.5 w-3.5"/> core/demographics • v2.7.0 • David Aranyi, 26m ago</span>
               </div>
             </div>
             <div className="flex items-center gap-3">
               <button className="flex items-center gap-2 bg-[#1a1a1a] hover:bg-[#222] border border-white/10 px-4 py-2 rounded-md text-sm text-white font-medium transition-colors">
                 <FileText className="h-4 w-4" /> Logs
               </button>
               <button className="bg-white text-black hover:bg-zinc-200 px-4 py-2 rounded-md text-sm font-semibold transition-colors">
                 Roll back ↗
               </button>
             </div>
           </div>

           {/* Stats Row */}
           <div className="grid grid-cols-4 gap-4 mb-8">
             <div className="bg-[#1a1a1a] border border-white/5 rounded-xl p-5">
               <div className="text-xs text-zinc-500 mb-2 font-medium">Requests</div>
               <div className="text-3xl font-bold text-white">1.3k/s</div>
             </div>
             <div className="bg-[#1a1a1a] border border-white/5 rounded-xl p-5">
               <div className="text-xs text-zinc-500 mb-2 font-medium">p95 latency</div>
               <div className="text-3xl font-bold text-white">402ms</div>
             </div>
             <div className="bg-[#1a1a1a] border border-white/5 rounded-xl p-5">
               <div className="text-xs text-zinc-500 mb-2 font-medium">Error rate</div>
               <div className="text-3xl font-bold text-white">1.9%</div>
             </div>
             <div className="bg-[#1a1a1a] border border-white/5 rounded-xl p-5">
               <div className="text-xs text-zinc-500 mb-2 font-medium">Saturation</div>
               <div className="text-3xl font-bold text-white">88%</div>
             </div>
           </div>

           {/* Moving Chart */}
           <div className="mb-12">
             <div className="flex justify-between items-end mb-4 border-b border-white/10 pb-2">
               <div className="text-sm font-semibold text-zinc-200">Throughput</div>
               <div className="text-xs text-zinc-500">last 40 min • <span className="text-white font-medium">1.3k/s</span></div>
             </div>
             <div className="relative w-full h-[280px] bg-transparent flex flex-col justify-between py-2">
                <div className="absolute inset-0 grid grid-rows-4 pointer-events-none">
                  <div className="border-b border-white/5"></div>
                  <div className="border-b border-white/5"></div>
                  <div className="border-b border-white/5"></div>
                  <div className="border-b border-white/5"></div>
                </div>
                
                {/* Y Axis Labels */}
                <div className="absolute left-0 top-0 h-full py-2 flex flex-col justify-between text-[10px] text-zinc-500 z-10 pointer-events-none -translate-x-1">
                  <span>2.0k</span>
                  <span>1.0k</span>
                  <span>0</span>
                </div>

                <svg className="absolute inset-0 h-full w-full pl-8" preserveAspectRatio="none" viewBox="0 0 100 100">
                  <defs>
                    <linearGradient id="chartGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#ffffff" stopOpacity="0.15" />
                      <stop offset="100%" stopColor="#ffffff" stopOpacity="0" />
                    </linearGradient>
                    <style>
                      {`
                        @keyframes moveRight {
                          from { stroke-dashoffset: 400; }
                          to { stroke-dashoffset: 0; }
                        }
                        .moving-path {
                          stroke-dasharray: 8 8;
                          animation: moveRight 80s linear infinite;
                        }
                      `}
                    </style>
                  </defs>
                  
                  {/* Fill Area */}
                  <path 
                    d="M 0 100 L 0 50 C 15 65, 30 35, 50 50 C 70 65, 85 45, 100 45 L 100 100 Z" 
                    fill="url(#chartGradient)" 
                  />
                  
                  {/* Moving Line representing path */}
                  <path 
                    d="M 0 50 C 15 65, 30 35, 50 50 C 70 65, 85 45, 100 45" 
                    fill="none" 
                    stroke="#ffffff" 
                    strokeWidth="1.5" 
                    className="moving-path"
                  />
                </svg>
             </div>
           </div>

           {/* Recent Activity */}
           <div>
             <h3 className="text-sm font-semibold text-zinc-200 mb-6">Recent activity</h3>
             <div className="space-y-6">
               
               <div className="flex items-start gap-4">
                  <div className="mt-1.5 h-1.5 w-1.5 rounded-full bg-amber-500 shrink-0" />
                  <div>
                    <h4 className="text-sm font-medium text-white">Write queue depth above threshold</h4>
                    <p className="text-[11px] text-zinc-500 mt-1">26m ago • saturation</p>
                  </div>
               </div>

               <div className="flex items-start gap-4">
                  <div className="mt-1.5 h-1.5 w-1.5 rounded-full bg-zinc-600 shrink-0" />
                  <div>
                    <h4 className="text-sm font-medium text-white">Shipped v2.7.0 with batched commits</h4>
                    <p className="text-[11px] text-zinc-500 mt-1">28m ago • deploy</p>
                  </div>
               </div>

               <div className="flex items-start gap-4">
                  <div className="mt-1.5 h-1.5 w-1.5 rounded-full bg-amber-500 shrink-0" />
                  <div>
                    <h4 className="text-sm font-medium text-white">Replica lag peaked at 4.2s</h4>
                    <p className="text-[11px] text-zinc-500 mt-1">1h ago • database</p>
                  </div>
               </div>

               <div className="flex items-start gap-4">
                  <div className="mt-1.5 h-1.5 w-1.5 rounded-full bg-zinc-600 shrink-0" />
                  <div>
                    <h4 className="text-sm font-medium text-white">Nightly reconciliation finished clean</h4>
                    <p className="text-[11px] text-zinc-500 mt-1">7h ago • job</p>
                  </div>
               </div>

             </div>
           </div>

        </div>
      </div>
      
      {/* Global CSS for scrollbar inside this file just for safety */}
      <style dangerouslySetInnerHTML={{__html: `
        .custom-scrollbar::-webkit-scrollbar {
          width: 6px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
          background: transparent; 
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: rgba(255, 255, 255, 0.1); 
          border-radius: 10px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover {
          background: rgba(255, 255, 255, 0.2); 
        }
      `}} />
    </div>
  );
}
