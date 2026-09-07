import React from 'react';
import { cn } from '../../utils/cn';

interface CardProps {
  children: React.ReactNode;
  className?: string;
  glow?: 'forecast' | 'critical' | 'warning' | 'safe' | false;
}

export function Card({ children, className, glow }: CardProps) {
  return (
    <div
      className={cn(
        'card',
        glow === 'forecast' && 'border-glow-forecast',
        glow === 'critical' && 'border-glow-critical',
        className
      )}
    >
      {children}
    </div>
  );
}

interface CardHeaderProps {
  title: string;
  subtitle?: string;
  right?: React.ReactNode;
  icon?: React.ReactNode;
}

export function CardHeader({ title, subtitle, right, icon }: CardHeaderProps) {
  return (
    <div className="card-header">
      <div className="flex items-center gap-2">
        {icon && <span className="text-forecast">{icon}</span>}
        <div>
          <h3 className="card-title">{title}</h3>
          {subtitle && <p className="text-[11px] text-text-muted mt-0.5">{subtitle}</p>}
        </div>
      </div>
      {right && <div className="flex items-center gap-2">{right}</div>}
    </div>
  );
}
