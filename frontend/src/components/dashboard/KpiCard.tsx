import React from 'react';
import { TrendingUp, TrendingDown, Minus } from 'lucide-react';
import { cn } from '../../utils/cn';
import { riskColorClass } from '../../utils/formatters';

interface KpiCardProps {
  title: string;
  value: string | number;
  unit?: string;
  delta?: number;
  deltaLabel?: string;
  severity?: 'critical' | 'warning' | 'elevated' | 'safe' | 'info' | 'forecast' | 'neutral';
  icon: React.ReactNode;
  subtitle?: string;
}

const bgMap: Record<string, string> = {
  critical: 'border-critical-dim',
  warning:  'border-warning-dim',
  elevated: 'border-elevated-dim',
  safe:     'border-safe-dim',
  info:     'border-info-dim',
  forecast: 'border-forecast-dim',
  neutral:  '',
};

const iconBgMap: Record<string, string> = {
  critical: 'bg-critical-bg text-critical',
  warning:  'bg-warning-bg  text-warning',
  elevated: 'bg-elevated-bg text-elevated',
  safe:     'bg-safe-bg     text-safe',
  info:     'bg-info-bg     text-info',
  forecast: 'bg-forecast-bg text-forecast',
  neutral:  'bg-surface-3   text-text-secondary',
};

export function KpiCard({
  title, value, unit, delta, deltaLabel, severity = 'neutral', icon, subtitle
}: KpiCardProps) {
  const deltaPositiveBad = severity === 'critical' || severity === 'warning' || severity === 'elevated';

  return (
    <div className={cn('kpi-card border', bgMap[severity])}>
      <div className="flex items-start justify-between">
        <div className={cn('w-8 h-8 rounded flex items-center justify-center flex-shrink-0', iconBgMap[severity])}>
          {icon}
        </div>
        {delta !== undefined && (
          <div className={cn(
            'flex items-center gap-0.5 text-[11px] font-medium',
            delta === 0 ? 'text-text-muted' :
            (delta > 0) === deltaPositiveBad ? 'text-critical' : 'text-safe'
          )}>
            {delta > 0 ? <TrendingUp size={11} /> : delta < 0 ? <TrendingDown size={11} /> : <Minus size={11} />}
            {Math.abs(delta)}{unit === '%' ? 'pp' : ''}
          </div>
        )}
      </div>
      <div>
        <div className="flex items-baseline gap-1">
          <span className="text-2xl font-bold text-text-primary tracking-tight">{value}</span>
          {unit && <span className="text-sm text-text-secondary">{unit}</span>}
        </div>
        <div className="text-[11px] font-medium text-text-secondary mt-0.5">{title}</div>
        {deltaLabel && <div className="text-[10px] text-text-muted">{deltaLabel}</div>}
        {subtitle && <div className="text-[10px] text-text-muted">{subtitle}</div>}
      </div>
    </div>
  );
}
