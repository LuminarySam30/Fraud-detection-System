import { ArrowRight, Boxes, Gauge, Layers, Scale, ShieldCheck } from "lucide-react";
import { PageHeader, Panel } from "@/components/ui";

const steps = [
  { icon: Boxes, title: "Ingest", detail: "1.3M transactions" },
  { icon: Layers, title: "Engineer", detail: "31 features" },
  { icon: Scale, title: "SMOTE balance", detail: "rare-class oversampling" },
  { icon: ShieldCheck, title: "XGBoost", detail: "500 trees" },
  { icon: Gauge, title: "Tune threshold", detail: "0.3029" },
];

const featureGroups = [
  { group: "Transaction", count: 6, examples: "amt, amt_ratio, amt_deviation, trans_hour" },
  { group: "Temporal", count: 4, examples: "trans_day_of_week, is_weekend, trans_month, unix_time" },
  { group: "Geographic", count: 2, examples: "distance_from_home, city_pop" },
  { group: "Customer", count: 3, examples: "age, gender_encoded, customer_avg_amt" },
  { group: "Merchant / category", count: 16, examples: "merchant_encoded, category_avg_amt, cat_* one-hots" },
];

const stack = [
  ["Model", "XGBoost (gradient-boosted trees)"],
  ["Class balancing", "SMOTE oversampling"],
  ["Serving API", "FastAPI on Render"],
  ["Frontend", "Next.js + Tailwind on Vercel"],
  ["Charts", "Plotly"],
];

export default function PipelinePage() {
  return (
    <div>
      <PageHeader
        eyebrow="Insights"
        title="Pipeline"
        description="How a raw transaction becomes a real-time risk score - from ingestion through to the served model."
      />

      <Panel className="mb-6 p-6">
        <h3 className="mb-5 text-lg font-bold text-fg">Training pipeline</h3>
        <div className="flex flex-col gap-3 sm:flex-row sm:items-stretch">
          {steps.map((s, i) => (
            <div key={s.title} className="flex flex-1 items-center gap-3 sm:flex-col sm:gap-3">
              <div className="flex w-full flex-1 flex-col items-center gap-2 rounded-xl border border-line bg-white/5 p-4 text-center">
                <div className="grid h-10 w-10 place-items-center rounded-lg bg-brand/10 text-brand"><s.icon className="h-5 w-5" aria-hidden /></div>
                <div className="text-sm font-semibold text-fg">{s.title}</div>
                <div className="text-xs text-fg-subtle">{s.detail}</div>
              </div>
              {i < steps.length - 1 && <ArrowRight className="hidden h-5 w-5 shrink-0 text-fg-subtle sm:block" aria-hidden />}
            </div>
          ))}
        </div>
      </Panel>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <Panel className="overflow-x-auto p-6 scrollbar-thin">
          <h3 className="mb-4 text-lg font-bold text-fg">Feature groups (31 total)</h3>
          <table className="w-full text-left text-sm">
            <thead className="border-b border-line bg-white/5 text-xs uppercase tracking-wide text-fg-subtle">
              <tr><th className="px-4 py-3">Group</th><th className="px-4 py-3">Count</th><th className="px-4 py-3">Examples</th></tr>
            </thead>
            <tbody>
              {featureGroups.map((g) => (
                <tr key={g.group} className="border-b border-line">
                  <td className="px-4 py-3 font-medium text-fg">{g.group}</td>
                  <td className="nums px-4 py-3 text-fg-muted">{g.count}</td>
                  <td className="nums px-4 py-3 text-xs text-fg-subtle">{g.examples}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Panel>

        <Panel className="p-6">
          <h3 className="mb-4 text-lg font-bold text-fg">Tech stack</h3>
          <dl className="divide-y divide-line">
            {stack.map(([k, v]) => (
              <div key={k} className="flex items-center justify-between gap-4 py-3 text-sm">
                <dt className="text-fg-muted">{k}</dt>
                <dd className="text-right font-medium text-fg">{v}</dd>
              </div>
            ))}
          </dl>
          <div className="mt-4 rounded-lg border border-warning/30 bg-warning/10 p-3 text-xs text-fg-muted">
            <span className="font-medium text-warning">Note on imbalance:</span> fraud is ~0.6% of transactions, so raw accuracy is misleading. The model is tuned for recall at a calibrated threshold, trading some precision to avoid missing fraud.
          </div>
        </Panel>
      </div>
    </div>
  );
}
