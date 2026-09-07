import React from 'react';
import { Card, CardHeader } from '../common/Card';
import { ProgressBar } from '../common/States';
import { Brain, FlaskConical } from 'lucide-react';
import type { mockExplainability } from '../../data/mockDashboard';

type Explainability = typeof mockExplainability;

export function ExplainabilityCard({ data }: { data: Explainability }) {
  return (
    <Card>
      <CardHeader
        title="Why This Forecast?"
        subtitle="Top factors influencing the prediction"
        icon={<Brain size={14} />}
        right={
          <span className="badge badge-forecast text-[9px]">AI Explainability</span>
        }
      />
      <div className="p-4 space-y-3">
        {data.factors.map((factor) => (
          <div key={factor.feature} className="space-y-1">
            <div className="flex items-center justify-between text-xs">
              <span className="text-text-secondary">{factor.feature}</span>
              <span className={`font-mono font-semibold ${factor.direction === 'positive' ? 'text-warning' : 'text-info'}`}>
                {factor.direction === 'positive' ? '+' : '−'}{factor.contribution}%
              </span>
            </div>
            <ProgressBar
              value={factor.contribution}
              color={factor.direction === 'positive' ? 'warning' : 'info'}
            />
          </div>
        ))}
        <div className="pt-2 border-t border-border-subtle">
          <div className="flex items-center gap-1.5 text-[10px] text-text-muted">
            <FlaskConical size={10} />
            <span>Demo Data — Simulated feature contribution estimates. Not real model output.</span>
          </div>
        </div>
      </div>
    </Card>
  );
}
