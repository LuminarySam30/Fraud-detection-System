import clsx from "clsx";
import { Loader2, type LucideIcon } from "lucide-react";

type Tone = "brand" | "accent" | "success" | "warning" | "danger" | "neutral";

const toneText: Record<Tone, string> = {
  brand: "text-brand",
  accent: "text-accent",
  success: "text-success",
  warning: "text-warning",
  danger: "text-danger",
  neutral: "text-fg-muted",
};

const toneBadge: Record<Tone, string> = {
  brand: "bg-brand/15 text-brand border-brand/30",
  accent: "bg-accent/15 text-accent border-accent/30",
  success: "bg-success/15 text-success border-success/30",
  warning: "bg-warning/15 text-warning border-warning/30",
  danger: "bg-danger/15 text-danger border-danger/30",
  neutral: "bg-white/5 text-fg-muted border-line",
};

export function Panel({
  className,
  hover,
  children,
}: {
  className?: string;
  hover?: boolean;
  children: React.ReactNode;
}) {
  return <div className={clsx("panel", hover && "panel-hover", className)}>{children}</div>;
}

export function PageHeader({
  eyebrow,
  title,
  description,
  actions,
}: {
  eyebrow?: string;
  title: string;
  description?: string;
  actions?: React.ReactNode;
}) {
  return (
    <header className="mb-8 flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
      <div>
        {eyebrow && (
          <div className="mb-2 font-mono text-xs font-medium uppercase tracking-widest text-brand">
            {eyebrow}
          </div>
        )}
        <h1 className="text-2xl font-bold text-fg sm:text-3xl">{title}</h1>
        {description && (
          <p className="mt-2 max-w-2xl text-sm leading-relaxed text-fg-muted">{description}</p>
        )}
      </div>
      {actions && <div className="flex shrink-0 flex-wrap items-center gap-2">{actions}</div>}
    </header>
  );
}

type ButtonProps = React.ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "primary" | "accent" | "ghost" | "subtle";
  loading?: boolean;
};

const buttonVariants: Record<NonNullable<ButtonProps["variant"]>, string> = {
  primary: "bg-brand text-[#04130d] hover:bg-accent shadow-lg shadow-brand/20",
  accent: "bg-accent text-[#04130d] hover:bg-brand font-semibold",
  ghost: "border border-line text-fg-muted hover:text-fg hover:border-line-strong",
  subtle: "bg-white/5 text-fg-muted hover:bg-white/10 hover:text-fg",
};

export function Button({
  variant = "primary",
  loading = false,
  className,
  children,
  disabled,
  ...props
}: ButtonProps) {
  return (
    <button
      className={clsx(
        "focus-ring inline-flex min-h-[44px] cursor-pointer items-center justify-center gap-2 rounded-lg px-4 py-2.5 text-sm font-medium transition disabled:pointer-events-none disabled:opacity-50",
        buttonVariants[variant],
        className,
      )}
      disabled={disabled || loading}
      {...props}
    >
      {loading && <Loader2 className="h-4 w-4 animate-spin" aria-hidden />}
      {children}
    </button>
  );
}

export function Badge({
  tone = "neutral",
  className,
  children,
}: {
  tone?: Tone;
  className?: string;
  children: React.ReactNode;
}) {
  return (
    <span
      className={clsx(
        "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs font-medium",
        toneBadge[tone],
        className,
      )}
    >
      {children}
    </span>
  );
}

export function StatCard({
  label,
  value,
  sub,
  icon: Icon,
  tone = "brand",
}: {
  label: string;
  value: React.ReactNode;
  sub?: React.ReactNode;
  icon?: LucideIcon;
  tone?: Tone;
}) {
  return (
    <Panel className="relative overflow-hidden p-5" hover>
      <div className="flex items-start justify-between gap-3">
        <span className="text-sm text-fg-muted">{label}</span>
        {Icon && <Icon className={clsx("h-5 w-5 shrink-0", toneText[tone])} aria-hidden />}
      </div>
      <div className="nums mt-3 text-2xl font-semibold text-fg">{value}</div>
      {sub && <div className="mt-1 text-sm text-fg-muted">{sub}</div>}
    </Panel>
  );
}

export function EmptyState({
  icon: Icon,
  title,
  description,
  action,
}: {
  icon?: LucideIcon;
  title: string;
  description?: string;
  action?: React.ReactNode;
}) {
  return (
    <Panel className="flex flex-col items-center justify-center gap-3 px-6 py-16 text-center">
      {Icon && (
        <div className="grid h-12 w-12 place-items-center rounded-xl bg-white/5 text-fg-subtle">
          <Icon className="h-6 w-6" aria-hidden />
        </div>
      )}
      <h3 className="text-lg font-semibold text-fg">{title}</h3>
      {description && (
        <p className="max-w-sm text-sm leading-relaxed text-fg-muted">{description}</p>
      )}
      {action && <div className="mt-2">{action}</div>}
    </Panel>
  );
}

export function ErrorState({
  title = "Couldn't load this view",
  message,
  onRetry,
  icon: Icon,
}: {
  title?: string;
  message?: string;
  onRetry?: () => void;
  icon?: LucideIcon;
}) {
  return (
    <Panel className="flex flex-col items-center justify-center gap-3 border-danger/30 bg-danger/5 px-6 py-16 text-center">
      {Icon && (
        <div className="grid h-12 w-12 place-items-center rounded-xl bg-danger/10 text-danger">
          <Icon className="h-6 w-6" aria-hidden />
        </div>
      )}
      <h3 className="text-lg font-semibold text-fg">{title}</h3>
      {message && <p className="max-w-md text-sm leading-relaxed text-fg-muted">{message}</p>}
      {onRetry && (
        <Button variant="ghost" onClick={onRetry} className="mt-2">
          Retry
        </Button>
      )}
    </Panel>
  );
}

export function Skeleton({ className }: { className?: string }) {
  return <div className={clsx("animate-pulse rounded-xl bg-white/5", className)} aria-hidden />;
}
