"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  BarChart3,
  Gauge,
  LayoutDashboard,
  Menu,
  ShieldCheck,
  Workflow,
  X,
} from "lucide-react";
import clsx from "clsx";

type NavItem = { name: string; href: string; icon: typeof Gauge };
type NavGroup = { label: string; items: NavItem[] };

const groups: NavGroup[] = [
  { label: "Overview", items: [{ name: "Home", href: "/", icon: LayoutDashboard }] },
  { label: "Live demo", items: [{ name: "Risk Scorecard", href: "/scorecard", icon: Gauge }] },
  {
    label: "Insights",
    items: [
      { name: "Model", href: "/model", icon: BarChart3 },
      { name: "Pipeline", href: "/pipeline", icon: Workflow },
    ],
  },
];

function Brand() {
  return (
    <Link href="/" className="focus-ring flex items-center gap-2.5 rounded-lg">
      <span className="grid h-8 w-8 place-items-center rounded-lg bg-gradient-to-tr from-brand to-accent shadow-[0_0_18px_rgba(16,185,129,0.45)]">
        <ShieldCheck className="h-4 w-4 text-[#04130d]" aria-hidden />
      </span>
      <span className="font-mono text-lg font-bold tracking-tight text-fg">
        Fraud<span className="text-gradient">Guard</span>
      </span>
    </Link>
  );
}

function NavLinks({ onNavigate }: { onNavigate?: () => void }) {
  const pathname = usePathname();
  return (
    <nav className="flex flex-col gap-5" aria-label="Primary">
      {groups.map((group) => (
        <div key={group.label}>
          <div className="mb-2 px-2 font-mono text-[10px] font-medium uppercase tracking-widest text-fg-subtle">
            {group.label}
          </div>
          <div className="flex flex-col gap-1">
            {group.items.map(({ name, href, icon: Icon }) => {
              const active = pathname === href;
              return (
                <Link
                  key={href}
                  href={href}
                  onClick={onNavigate}
                  aria-current={active ? "page" : undefined}
                  className={clsx(
                    "focus-ring flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition",
                    active
                      ? "border border-brand/30 bg-brand/15 text-brand"
                      : "border border-transparent text-fg-muted hover:bg-white/5 hover:text-fg",
                  )}
                >
                  <Icon className={clsx("h-4 w-4 shrink-0", active ? "text-brand" : "text-fg-subtle")} aria-hidden />
                  {name}
                </Link>
              );
            })}
          </div>
        </div>
      ))}
    </nav>
  );
}

function Footer() {
  return (
    <div className="border-t border-line px-4 py-4 text-center">
      <div className="font-mono text-[10px] font-semibold uppercase tracking-widest text-brand/80">
        Built by Samuel Jeromiah
      </div>
      <div className="mt-0.5 text-xs text-fg-subtle">FraudGuard · XGBoost</div>
    </div>
  );
}

export default function Sidebar() {
  const [open, setOpen] = useState(false);
  const pathname = usePathname();

  useEffect(() => {
    setOpen(false);
  }, [pathname]);

  useEffect(() => {
    document.body.style.overflow = open ? "hidden" : "";
    return () => {
      document.body.style.overflow = "";
    };
  }, [open]);

  return (
    <>
      <aside className="fixed inset-y-0 left-0 z-40 hidden w-64 flex-col border-r border-line bg-surface/70 backdrop-blur-xl lg:flex">
        <div className="border-b border-line p-5">
          <Brand />
        </div>
        <div className="flex-1 overflow-y-auto scrollbar-thin p-4">
          <NavLinks />
        </div>
        <Footer />
      </aside>

      <header className="fixed inset-x-0 top-0 z-40 flex h-16 items-center justify-between border-b border-line bg-surface/80 px-4 backdrop-blur-xl lg:hidden">
        <Brand />
        <button
          onClick={() => setOpen(true)}
          aria-label="Open navigation menu"
          aria-expanded={open}
          className="focus-ring grid h-11 w-11 cursor-pointer place-items-center rounded-lg text-fg-muted hover:bg-white/5 hover:text-fg"
        >
          <Menu className="h-5 w-5" aria-hidden />
        </button>
      </header>

      {open && (
        <div className="fixed inset-0 z-50 lg:hidden">
          <button
            aria-label="Close navigation menu"
            onClick={() => setOpen(false)}
            className="absolute inset-0 bg-black/60 backdrop-blur-sm"
          />
          <div className="absolute inset-y-0 left-0 flex w-72 max-w-[85%] flex-col border-r border-line bg-surface shadow-2xl">
            <div className="flex items-center justify-between border-b border-line p-5">
              <Brand />
              <button
                onClick={() => setOpen(false)}
                aria-label="Close navigation menu"
                className="focus-ring grid h-11 w-11 cursor-pointer place-items-center rounded-lg text-fg-muted hover:bg-white/5 hover:text-fg"
              >
                <X className="h-5 w-5" aria-hidden />
              </button>
            </div>
            <div className="flex-1 overflow-y-auto scrollbar-thin p-4">
              <NavLinks onNavigate={() => setOpen(false)} />
            </div>
            <Footer />
          </div>
        </div>
      )}
    </>
  );
}
