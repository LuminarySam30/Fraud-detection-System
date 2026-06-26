"use client";

/* eslint-disable @typescript-eslint/no-explicit-any */
import dynamic from "next/dynamic";

// Lazy-load Plotly (it touches `window`, so it can't render on the server).
const PlotlyChart = dynamic(() => import("react-plotly.js"), {
  ssr: false,
  loading: () => (
    <div className="flex h-full min-h-[200px] w-full animate-pulse items-center justify-center rounded-xl bg-white/5 text-sm text-fg-subtle">
      Loading chart...
    </div>
  ),
});

interface PlotProps {
  data: any[];
  layout?: any;
  config?: any;
  className?: string;
}

// Shared dark theme matching the FraudGuard tokens (emerald / mint / status).
const AXIS = {
  gridcolor: "rgba(167,243,208,0.10)",
  zerolinecolor: "rgba(167,243,208,0.22)",
  linecolor: "rgba(167,243,208,0.22)",
  tickfont: { size: 11 },
};

export default function Plot({ data, layout = {}, config = {}, className }: PlotProps) {
  const themedLayout = {
    paper_bgcolor: "transparent",
    plot_bgcolor: "transparent",
    font: { family: "IBM Plex Sans, sans-serif", color: "#a7c3b8", size: 12 },
    colorway: ["#10b981", "#34d399", "#22c55e", "#fbbf24", "#fb7185", "#6ee7b7"],
    margin: { t: 24, r: 16, l: 48, b: 40 },
    hoverlabel: { bgcolor: "#14201b", bordercolor: "rgba(167,243,208,0.22)", font: { family: "IBM Plex Sans" } },
    ...layout,
    // applied after ...layout so per-call axes still inherit the themed defaults
    xaxis: { ...AXIS, ...(layout.xaxis ?? {}) },
    yaxis: { ...AXIS, ...(layout.yaxis ?? {}) },
    legend: { font: { size: 11 }, ...(layout.legend ?? {}) },
  };

  const themedConfig = { displayModeBar: false, responsive: true, ...config };

  return (
    <div className={`h-full w-full ${className ?? ""}`}>
      <PlotlyChart
        data={data}
        layout={themedLayout}
        config={themedConfig}
        useResizeHandler
        style={{ width: "100%", height: "100%" }}
      />
    </div>
  );
}
