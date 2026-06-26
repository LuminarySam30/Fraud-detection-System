"use client";

import { useEffect, useState } from "react";
import { AlertTriangle, Beaker, CheckCircle2, Gauge as GaugeIcon, ShieldAlert } from "lucide-react";
import { apiFetch, ApiError } from "@/lib/api";
import { Badge, Button, ErrorState, PageHeader, Panel } from "@/components/ui";

const FALLBACK_CATEGORIES = [
  "entertainment", "food_dining", "gas_transport", "grocery_net", "grocery_pos",
  "health_fitness", "home", "kids_pets", "misc_net", "misc_pos",
  "personal_care", "shopping_net", "shopping_pos", "travel",
];
const DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"];

interface Meta { threshold: number; categories: string[] }
interface ScoreResult {
  probability: number;
  threshold: number;
  is_fraud: boolean;
  risk_tier: "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";
  confidence: number;
  factors: { factor: string; value: string; signal: string }[];
}

const SAMPLE = {
  amount: 925, category: "grocery_pos", hour: 2, day_of_week: 6, distance_from_home: 280,
  age: 81, gender: "Female", customer_avg_amt: 48, merchant_risk: 0.8, city_pop: 1200, is_new_merchant: true,
};
const SAFE = {
  amount: 42, category: "food_dining", hour: 13, day_of_week: 2, distance_from_home: 4,
  age: 34, gender: "Male", customer_avg_amt: 55, merchant_risk: 0.3, city_pop: 80000, is_new_merchant: false,
};

const tierTone = { LOW: "success", MEDIUM: "warning", HIGH: "danger", CRITICAL: "danger" } as const;

export default function ScorecardPage() {
  const [categories, setCategories] = useState<string[]>(FALLBACK_CATEGORIES);
  const [form, setForm] = useState({ ...SAMPLE });
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ScoreResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    apiFetch<Meta>("/api/metadata")
      .then((m) => m.categories?.length && setCategories(m.categories))
      .catch(() => {});
  }, []);

  const set = (k: string, v: string | number | boolean) => setForm((f) => ({ ...f, [k]: v }));

  const submit = async (e?: React.FormEvent) => {
    e?.preventDefault();
    setLoading(true);
    setError(null);
    try {
      setResult(await apiFetch<ScoreResult>("/api/score", { method: "POST", body: JSON.stringify(form) }));
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Scoring failed.");
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  const inputCls =
    "focus-ring min-h-[44px] w-full rounded-lg border border-line bg-white/5 px-3 py-2 text-sm text-fg placeholder:text-fg-subtle/70 focus:border-brand/50";
  const labelCls = "mb-1 block text-xs font-medium uppercase tracking-wide text-fg-muted";

  return (
    <div>
      <PageHeader
        eyebrow="Live demo · scored by the real model"
        title="Risk Scorecard"
        description="Enter a transaction and the trained XGBoost model returns a fraud probability, a threshold-based decision, and the signals behind it. Nothing is mocked - this calls the model directly."
        actions={
          <>
            <Button variant="subtle" type="button" onClick={() => { setForm({ ...SAMPLE }); setResult(null); setError(null); }}>
              <Beaker className="h-4 w-4" aria-hidden /> Risky sample
            </Button>
            <Button variant="ghost" type="button" onClick={() => { setForm({ ...SAFE }); setResult(null); setError(null); }}>
              Safe sample
            </Button>
          </>
        }
      />

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <Panel className="p-6 lg:col-span-2">
          <form onSubmit={submit} className="space-y-6">
            <div className="grid grid-cols-1 gap-8 sm:grid-cols-2">
              {/* Transaction */}
              <fieldset className="space-y-4">
                <legend className="mb-1 w-full border-b border-line pb-2 font-mono text-xs font-semibold uppercase tracking-widest text-brand">
                  Transaction
                </legend>
                <div>
                  <label htmlFor="amount" className={labelCls}>Amount ($)</label>
                  <input id="amount" type="number" min={0} step="any" value={form.amount} onChange={(e) => set("amount", parseFloat(e.target.value) || 0)} className={inputCls} />
                </div>
                <div>
                  <label htmlFor="category" className={labelCls}>Category</label>
                  <select id="category" value={form.category} onChange={(e) => set("category", e.target.value)} className={inputCls}>
                    {categories.map((c) => <option key={c} value={c}>{c.replace(/_/g, " ")}</option>)}
                  </select>
                </div>
                <div>
                  <label htmlFor="hour" className={labelCls}>Hour of day · {String(form.hour).padStart(2, "0")}:00</label>
                  <input id="hour" type="range" min={0} max={23} step={1} value={form.hour} onChange={(e) => set("hour", parseInt(e.target.value))} className="h-2 w-full cursor-pointer accent-[var(--color-brand)]" />
                </div>
                <div>
                  <label htmlFor="dow" className={labelCls}>Day of week</label>
                  <select id="dow" value={form.day_of_week} onChange={(e) => set("day_of_week", parseInt(e.target.value))} className={inputCls}>
                    {DAYS.map((d, i) => <option key={d} value={i}>{d}</option>)}
                  </select>
                </div>
                <div>
                  <label htmlFor="distance" className={labelCls}>Distance from home (mi)</label>
                  <input id="distance" type="number" min={0} step="any" value={form.distance_from_home} onChange={(e) => set("distance_from_home", parseFloat(e.target.value) || 0)} className={inputCls} />
                </div>
              </fieldset>

              {/* Customer */}
              <fieldset className="space-y-4">
                <legend className="mb-1 w-full border-b border-line pb-2 font-mono text-xs font-semibold uppercase tracking-widest text-accent">
                  Customer & merchant
                </legend>
                <div>
                  <label htmlFor="avg" className={labelCls}>Typical spend ($)</label>
                  <input id="avg" type="number" min={0} step="any" value={form.customer_avg_amt} onChange={(e) => set("customer_avg_amt", parseFloat(e.target.value) || 0)} className={inputCls} />
                </div>
                <div>
                  <label htmlFor="age" className={labelCls}>Age</label>
                  <input id="age" type="number" min={14} max={100} step={1} value={form.age} onChange={(e) => set("age", parseInt(e.target.value) || 0)} className={inputCls} />
                </div>
                <div>
                  <label htmlFor="merch" className={labelCls}>Merchant risk · {form.merchant_risk.toFixed(2)}</label>
                  <input id="merch" type="range" min={0} max={1} step={0.01} value={form.merchant_risk} onChange={(e) => set("merchant_risk", parseFloat(e.target.value))} className="h-2 w-full cursor-pointer accent-[var(--color-accent)]" />
                </div>
                <div>
                  <label htmlFor="pop" className={labelCls}>City population</label>
                  <input id="pop" type="number" min={0} step="any" value={form.city_pop} onChange={(e) => set("city_pop", parseInt(e.target.value) || 0)} className={inputCls} />
                </div>
                <div className="flex items-center gap-6 pt-1">
                  <span className={labelCls + " mb-0"}>Gender</span>
                  <label className="flex items-center gap-2 text-sm text-fg"><input type="radio" name="gender" checked={form.gender === "Male"} onChange={() => set("gender", "Male")} className="accent-[var(--color-brand)]" /> Male</label>
                  <label className="flex items-center gap-2 text-sm text-fg"><input type="radio" name="gender" checked={form.gender === "Female"} onChange={() => set("gender", "Female")} className="accent-[var(--color-brand)]" /> Female</label>
                </div>
                <label className="flex items-center gap-2 text-sm text-fg-muted">
                  <input type="checkbox" checked={form.is_new_merchant} onChange={(e) => set("is_new_merchant", e.target.checked)} className="h-4 w-4 accent-[var(--color-brand)]" />
                  First time at this merchant
                </label>
              </fieldset>
            </div>
            <Button type="submit" loading={loading} className="w-full">
              {!loading && <GaugeIcon className="h-4 w-4" aria-hidden />}
              {loading ? "Scoring..." : "Score transaction"}
            </Button>
          </form>
        </Panel>

        <div className="lg:col-span-1">
          {error ? (
            <ErrorState title="Scoring failed" message={error} icon={AlertTriangle} onRetry={() => submit()} />
          ) : result ? (
            <Panel className="flex flex-col items-center gap-5 p-6">
              <Gauge probability={result.probability} tier={result.risk_tier} />
              {result.is_fraud ? (
                <div className="flex w-full items-center justify-center gap-2 rounded-lg border border-danger/30 bg-danger/10 py-2 text-sm font-semibold text-danger">
                  <ShieldAlert className="h-4 w-4" aria-hidden /> FRAUD ALERT
                </div>
              ) : (
                <div className="flex w-full items-center justify-center gap-2 rounded-lg border border-success/30 bg-success/10 py-2 text-sm font-semibold text-success">
                  <CheckCircle2 className="h-4 w-4" aria-hidden /> APPROVED
                </div>
              )}
              <div className="grid w-full grid-cols-2 gap-3 text-center">
                <div className="rounded-lg bg-white/5 p-3">
                  <div className="text-xs text-fg-subtle">Risk tier</div>
                  <div className="mt-1"><Badge tone={tierTone[result.risk_tier]}>{result.risk_tier}</Badge></div>
                </div>
                <div className="rounded-lg bg-white/5 p-3">
                  <div className="text-xs text-fg-subtle">Threshold</div>
                  <div className="nums mt-1 font-semibold text-fg">{(result.threshold * 100).toFixed(1)}%</div>
                </div>
              </div>
              <div className="w-full space-y-2.5 border-t border-line pt-4">
                <div className="font-mono text-[10px] font-medium uppercase tracking-widest text-fg-subtle">Risk factors</div>
                {result.factors.map((f) => (
                  <div key={f.factor} className="flex items-center justify-between gap-3 text-sm">
                    <span className="text-fg-muted">{f.factor}</span>
                    <span className="flex items-center gap-2">
                      <span className="nums text-fg">{f.value}</span>
                      <span className={`h-2 w-2 rounded-full ${f.signal === "high" ? "bg-danger" : f.signal === "medium" ? "bg-warning" : "bg-success"}`} aria-label={f.signal} />
                    </span>
                  </div>
                ))}
              </div>
            </Panel>
          ) : (
            <Panel className="flex h-full min-h-[320px] flex-col items-center justify-center gap-4 p-6 text-center">
              <div className="grid h-14 w-14 place-items-center rounded-2xl bg-white/5 text-fg-subtle">
                <GaugeIcon className="h-7 w-7" aria-hidden />
              </div>
              <p className="max-w-[16rem] text-sm leading-relaxed text-fg-muted">
                Adjust the transaction (or load a sample) and score it to see the fraud probability and the factors driving it.
              </p>
            </Panel>
          )}
        </div>
      </div>
    </div>
  );
}

function Gauge({ probability, tier }: { probability: number; tier: string }) {
  const pct = Math.round(probability * 100);
  const r = 56;
  const c = 2 * Math.PI * r;
  const offset = c * (1 - probability);
  const color = tier === "LOW" ? "var(--color-success)" : tier === "MEDIUM" ? "var(--color-warning)" : "var(--color-danger)";
  return (
    <div className="relative h-36 w-36" role="img" aria-label={`Fraud probability ${pct} percent`}>
      <svg viewBox="0 0 128 128" className="h-full w-full -rotate-90">
        <circle cx="64" cy="64" r={r} fill="none" stroke="rgba(148,163,184,0.15)" strokeWidth="10" />
        <circle cx="64" cy="64" r={r} fill="none" stroke={color} strokeWidth="10" strokeLinecap="round" strokeDasharray={c} strokeDashoffset={offset} style={{ transition: "stroke-dashoffset 0.6s ease" }} />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="nums text-3xl font-bold text-fg">{pct}%</span>
        <span className="text-[10px] uppercase tracking-widest text-fg-subtle">fraud risk</span>
      </div>
    </div>
  );
}
