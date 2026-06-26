import type { Metadata } from "next";
import { IBM_Plex_Sans, IBM_Plex_Mono } from "next/font/google";
import "./globals.css";
import Sidebar from "@/components/Sidebar";
import BackendStatus from "@/components/BackendStatus";

const plexSans = IBM_Plex_Sans({
  subsets: ["latin"],
  weight: ["300", "400", "500", "600", "700"],
  variable: "--font-plex-sans",
  display: "swap",
});

const plexMono = IBM_Plex_Mono({
  subsets: ["latin"],
  weight: ["400", "500", "600"],
  variable: "--font-plex-mono",
  display: "swap",
});

export const metadata: Metadata = {
  title: "FraudGuard - Real-Time Transaction Risk Scoring",
  description:
    "An XGBoost credit-card fraud classifier (31 engineered features) served as a live risk-scoring API with an interactive scorecard and model dashboard.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={`${plexSans.variable} ${plexMono.variable}`}>
      <body className="antialiased">
        <a
          href="#main"
          className="focus-ring sr-only z-[100] rounded-lg bg-brand px-4 py-2 text-sm font-medium text-[#04130d] focus:not-sr-only focus:absolute focus:left-4 focus:top-4"
        >
          Skip to content
        </a>
        <Sidebar />
        <div className="lg:pl-64">
          <main
            id="main"
            className="mx-auto min-h-dvh max-w-7xl px-4 pb-16 pt-20 sm:px-6 lg:px-8 lg:pt-8"
          >
            <BackendStatus />
            {children}
          </main>
        </div>
      </body>
    </html>
  );
}
