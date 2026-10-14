import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { Providers } from "@/app/providers";
import { TooltipProvider } from "@/components/ui/tooltip";
import { AppShell } from "@/components/shared/AppShell";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Intelligent Pattern Discovery",
  description: "Explainable AI Analytics Engine",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className={`${inter.className} bg-background text-foreground min-h-screen flex`}>
        <TooltipProvider>
          <Providers>
            <AppShell>
              {children}
            </AppShell>
          </Providers>
        </TooltipProvider>
      </body>
    </html>
  );
}
