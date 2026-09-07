import React from 'react';
import { cn } from '../../utils/cn';

type Severity = 'Critical' | 'High' | 'Medium' | 'Low' | 'Safe' | 'Info' | 'Forecast' | 'Muted';

interface BadgeProps {
  severity?: Severity;
  children: React.ReactNode;
  className?: string;
  dot?: boolean;
}

const variantMap: Record<Severity, string> = {
  Critical: 'badge-critical',
  High:     'badge-warning',
  Medium:   'badge-elevated',
  Low:      'badge-info',
  Safe:     'badge-safe',
  Info:     'badge-info',
  Forecast: 'badge-forecast',
  Muted:    'badge-muted',
};

const dotColor: Record<Severity, string> = {
  Critical: 'bg-critical',
  High:     'bg-warning',
  Medium:   'bg-elevated',
  Low:      'bg-info',
  Safe:     'bg-safe',
  Info:     'bg-info',
  Forecast: 'bg-forecast',
  Muted:    'bg-text-muted',
};

export function Badge({ severity = 'Muted', children, className, dot = false }: BadgeProps) {
  return (
    <span className={cn(variantMap[severity], className)}>
      {dot && <span className={cn('w-1.5 h-1.5 rounded-full', dotColor[severity])} />}
      {children}
    </span>
  );
}

export function RiskBadge({ score }: { score: number }) {
  const severity: Severity =
    score >= 80 ? 'Critical' :
    score >= 65 ? 'High'     :
    score >= 45 ? 'Medium'   :
    score >= 25 ? 'Low'      : 'Safe';
  const label =
    score >= 80 ? 'CRITICAL' :
    score >= 65 ? 'HIGH'     :
    score >= 45 ? 'MEDIUM'   :
    score >= 25 ? 'LOW'      : 'SAFE';
  return <Badge severity={severity} dot>{label} · {score}%</Badge>;
}

export function StageBadge({ stage }: { stage: string }) {
  const isObserved = stage === 'observed';
  const isForecast = stage === 'forecast';
  return (
    <Badge severity={isForecast ? 'Forecast' : isObserved ? 'Safe' : 'Muted'}>
      {isForecast ? '◆ FORECAST' : isObserved ? '✓ OBSERVED' : stage}
    </Badge>
  );
}
