import React from 'react';
import { cn } from '../../utils/cn';

// ── Skeleton ───────────────────────────────────────────
export function Skeleton({ className }: { className?: string }) {
  return <div className={cn('skeleton', className)} />;
}

export function SkeletonCard({ rows = 4 }: { rows?: number }) {
  return (
    <div className="card p-4 space-y-3">
      <Skeleton className="h-4 w-1/3" />
      {Array.from({ length: rows }).map((_, i) => (
        <Skeleton key={i} className={cn('h-3', i % 2 === 0 ? 'w-full' : 'w-4/5')} />
      ))}
    </div>
  );
}

export function SkeletonTable({ rows = 6 }: { rows?: number }) {
  return (
    <div className="space-y-2 p-3">
      <Skeleton className="h-6 w-full" />
      {Array.from({ length: rows }).map((_, i) => (
        <Skeleton key={i} className="h-8 w-full opacity-70" />
      ))}
    </div>
  );
}

// ── Empty State ────────────────────────────────────────
interface EmptyStateProps {
  icon?: React.ReactNode;
  title: string;
  description?: string;
  action?: React.ReactNode;
}

export function EmptyState({ icon, title, description, action }: EmptyStateProps) {
  return (
    <div className="flex flex-col items-center justify-center py-16 px-6 text-center gap-3">
      {icon && <div className="text-text-disabled text-3xl">{icon}</div>}
      <p className="text-text-secondary font-medium text-sm">{title}</p>
      {description && <p className="text-text-muted text-xs max-w-xs">{description}</p>}
      {action && <div className="mt-2">{action}</div>}
    </div>
  );
}

// ── Error State ────────────────────────────────────────
interface ErrorStateProps {
  message?: string;
  onRetry?: () => void;
}

export function ErrorState({ message, onRetry }: ErrorStateProps) {
  return (
    <div className="flex flex-col items-center justify-center py-12 px-6 text-center gap-3">
      <div className="text-critical text-2xl">⚠</div>
      <p className="text-text-secondary text-sm font-medium">Unable to load data</p>
      <p className="text-text-muted text-xs max-w-xs">
        {message || 'The server is unavailable. Running in demo mode with synthetic data.'}
      </p>
      {onRetry && (
        <button onClick={onRetry} className="btn btn-secondary btn-sm mt-1">
          Retry
        </button>
      )}
    </div>
  );
}

// ── Loading Spinner ────────────────────────────────────
export function Spinner({ size = 16 }: { size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      className="animate-spin text-forecast"
    >
      <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2" strokeOpacity="0.2" />
      <path d="M12 2a10 10 0 0 1 10 10" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
    </svg>
  );
}

// ── Progress Bar ───────────────────────────────────────
interface ProgressBarProps {
  value: number; // 0-100
  color?: 'forecast' | 'critical' | 'warning' | 'safe' | 'info';
  className?: string;
  showLabel?: boolean;
}

const colorMap = {
  forecast: 'bg-forecast',
  critical: 'bg-critical',
  warning:  'bg-warning',
  safe:     'bg-safe',
  info:     'bg-info',
};

export function ProgressBar({ value, color = 'forecast', className, showLabel }: ProgressBarProps) {
  return (
    <div className="flex items-center gap-2">
      <div className={cn('progress-bar flex-1', className)}>
        <div
          className={cn('progress-fill', colorMap[color])}
          style={{ width: `${Math.min(100, Math.max(0, value))}%` }}
        />
      </div>
      {showLabel && <span className="text-[11px] text-text-muted w-8 text-right">{value}%</span>}
    </div>
  );
}
