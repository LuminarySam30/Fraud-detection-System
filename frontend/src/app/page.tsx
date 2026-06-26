import Link from "next/link";
import {
  ArrowRight,
  BarChart3,
  Boxes,
  Gauge,
  Layers,
  ScanSearch,
  ShieldCheck,
  Workflow,
} from "lucide-react";
import { Panel } from "@/components/ui";

const stats = [
  { label: "Engineered features", value: "31" },
  { label: "Model", value: "XGBoost" },
  { label: "Decision threshold", value: "0.30" },
  { label: "Transaction categories", value: "14" },
];

const steps = [
  { icon: Boxes, title: "Ingest", body: "1.3M+ historical credit-card transactions, heavily imbalanced (~0.6% fraud)." },
  { icon: Layers, title: "Engineer", body: "31 features - amount ratios, distance-from-home, time-of-day, target-encoded merchant/category." },
  { icon: ShieldCheck, title: "Train", body: "XGBoost with SMOTE balancing; the decision threshold is tuned (0.30), not left at 0.5." },
  { icon: Gauge, title: "Score", body: "Any new transaction is scored in real time and assigned a risk tier." },
];

const features = [
  { icon: ScanSearch, title: "Live risk scoring", body: "Enter a transaction and get an instant fraud probability, decision, and per-factor breakdown - served straight from the trained model." },
  { icon: BarChart3, title: "Honest feature importance", body: "The Model page reads importances directly from the live XGBoost model, so what you see is what actually drives predictions." },
  { icon: ShieldCheck, title: "Threshold-aware", body: "Fraud is rare, so accuracy is misleading. The model optimises recall at a tuned threshold to catch fraud without drowning analysts in alerts." },
  { icon: Workflow, title: "Reproducible pipeline", body: "From raw transactions through feature engineering, balancing, and evaluation - the whole path is documented." },
];

const tech = ["Next.js 16", "Tailwind v4", "FastAPI", "XGBoost", "scikit-learn", "Plotly"];

export default function Home() {
  return (
    <div className="space-y-16 pb-8">
      <section className="pt-6 sm:pt-10">
        <div className="mb-5 inline-flex items-center gap-2 rounded-full border border-line bg-white/5 px-3 py-1 font-mono text-xs text-fg-muted">
          <span className="h-1.5 w-1.5 rounded-full bg-success" /> Fraud Detection · XGBoost + FastAPI
        </div>
        <h1 className="max-w-3xl text-4xl font-bold leading-[1.1] text-fg sm:text-5xl">
          Catch fraudulent transactions <span className="text-gradient">before they clear.</span>
        </h1>
        <p className="mt-5 max-w-2xl text-lg leading-relaxed text-fg-muted">
          FraudGuard scores credit-card transactions in real time with a gradient-boosted model trained on
          1.3M+ records - turning 31 engineered signals into a single, explainable risk decision.
        </p>
        <div className="mt-8 flex flex-wrap items-center gap-3">
          <Link
            href="/scorecard"
            className="focus-ring inline-flex min-h-[44px] items-center gap-2 rounded-lg bg-brand px-6 py-3 text-sm font-semibold text-[#04130d] shadow-lg shadow-brand/20 transition hover:bg-accent"
          >
            <Gauge className="h-4 w-4" aria-hidden /> Try the Risk Scorecard
          </Link>
          <Link
            href="/model"
            className="focus-ring inline-flex min-h-[44px] items-center gap-2 rounded-lg border border-line px-6 py-3 text-sm font-medium text-fg-muted transition hover:border-line-strong hover:text-fg"
          >
            Explore the model <ArrowRight className="h-4 w-4" aria-hidden />
          </Link>
        </div>
      </section>

      <section className="grid grid-cols-2 gap-3 sm:gap-4 lg:grid-cols-4">
        {stats.map((s) => (
          <Panel key={s.label} className="p-5" hover>
            <div className="nums text-2xl font-semibold text-fg">{s.value}</div>
            <div className="mt-1 text-sm text-fg-muted">{s.label}</div>
          </Panel>
        ))}
      </section>

      <section>
        <h2 className="mb-6 text-xl font-bold text-fg">How it works</h2>
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {steps.map((step, i) => (
            <Panel key={step.title} className="relative p-6" hover>
              <span className="nums absolute right-5 top-5 text-sm text-fg-subtle">0{i + 1}</span>
              <div className="mb-4 grid h-11 w-11 place-items-center rounded-xl bg-brand/10 text-brand">
                <step.icon className="h-5 w-5" aria-hidden />
              </div>
              <h3 className="font-semibold text-fg">{step.title}</h3>
              <p className="mt-1.5 text-sm leading-relaxed text-fg-muted">{step.body}</p>
            </Panel>
          ))}
        </div>
      </section>

      <section>
        <h2 className="mb-6 text-xl font-bold text-fg">What makes it trustworthy</h2>
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
          {features.map((f) => (
            <Panel key={f.title} className="flex gap-4 p-6" hover>
              <div className="grid h-11 w-11 shrink-0 place-items-center rounded-xl bg-accent/10 text-accent">
                <f.icon className="h-5 w-5" aria-hidden />
              </div>
              <div>
                <h3 className="font-semibold text-fg">{f.title}</h3>
                <p className="mt-1.5 text-sm leading-relaxed text-fg-muted">{f.body}</p>
              </div>
            </Panel>
          ))}
        </div>
      </section>

      <section className="flex flex-wrap items-center gap-2">
        <span className="mr-2 text-sm text-fg-subtle">Built with</span>
        {tech.map((t) => (
          <span key={t} className="nums rounded-full border border-line bg-white/5 px-3 py-1 text-xs text-fg-muted">
            {t}
          </span>
        ))}
      </section>
    </div>
  );
}
