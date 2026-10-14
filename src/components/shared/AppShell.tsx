"use client";

import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { usePathname } from "next/navigation";
import { Sidebar } from "./Sidebar";
import { Loader2 } from "lucide-react";

export function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const isLanding = pathname === "/";
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    // Only show loader when navigating between dashboard/exploration pages, not on initial load of landing page
    if (!isLanding) {
      setIsLoading(true);
      const timer = setTimeout(() => {
        setIsLoading(false);
      }, 2000);
      return () => clearTimeout(timer);
    }
  }, [pathname, isLanding]);

  return (
    <div className={`flex w-full min-h-screen ${isLanding ? 'bg-[#050505]' : 'bg-background'}`}>
      <Sidebar />
      <main className="flex-1 flex flex-col min-h-screen overflow-hidden relative">
        <AnimatePresence mode="wait">
          {isLoading ? (
            <motion.div
              key="loader"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="absolute inset-0 z-50 flex flex-col items-center justify-center bg-background/80 backdrop-blur-md"
            >
              <Loader2 className="h-12 w-12 animate-spin text-indigo-500 mb-4" />
              <p className="text-lg font-medium text-muted-foreground animate-pulse">Loading analytical intelligence...</p>
            </motion.div>
          ) : (
            <motion.div
              key={pathname}
              initial={isLanding ? { opacity: 1 } : { opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              transition={{ duration: 0.3, ease: "easeOut" }}
              className={`flex-1 overflow-auto ${!isLanding ? 'p-6 md:p-8' : ''}`}
            >
              {children}
            </motion.div>
          )}
        </AnimatePresence>
      </main>
    </div>
  );
}
