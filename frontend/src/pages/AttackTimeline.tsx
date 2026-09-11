import React from 'react';
import { AppShell } from '../components/layout/AppShell';
import { Card, CardHeader } from '../components/common/Card';
import { Badge } from '../components/common/Badge';
import { useAnalysis } from '../hooks/useAnalysis';
import { GitBranch, Check, Zap, Clock } from 'lucide-react';
import { cn } from '../utils/cn';

export default function AttackTimeline() {
  const { analysis } = useAnalysis();

  if (!analysis) {
    return (
      <AppShell>
        <div className="flex flex-col items-center justify-center h-full min-h-[60vh] text-center">
          <Clock size={48} className="text-text-disabled mb-4" />
          <h2 className="text-xl font-bold text-text-primary mb-2">No Dataset Uploaded</h2>
          <p className="text-text-muted">Upload a dataset to view the timeline of observed and predicted events.</p>
        </div>
      </AppShell>
    );
  }

  // Map directly from unified forecast array
  const events = (analysis.forecast || []).map((f: any) => ({
    id: f.step,
    observed: f.step === 0, // step 0 is NOW (Observed)
    stage: f.stage || 'Unknown',
    label: f.mitre_tactic && f.mitre_technique ? `${f.mitre_tactic} (${f.mitre_technique})` : (f.state || 'Unknown Activity'),
    detail: f.step === 0 ? 'Confirmed network state' : `Predicted trajectory at t+${f.step}`,
    riskScore: f.risk || 0,
  }));

  return (
    <AppShell>
      <div className="mb-4">
        <h2 className="text-xl font-bold text-text-primary tracking-tight">Attack Timeline</h2>
        <p className="text-sm text-text-muted mt-1">Chronological sequence of observed network events and predicted future actions.</p>
      </div>

      <Card>
        <CardHeader
          title="Event Timeline"
          icon={<GitBranch size={14} />}
          right={
            <div className="flex items-center gap-3 text-[10px]">
              <div className="flex items-center gap-1"><div className="w-2 h-2 rounded-full bg-safe" /><span>Observed Evidence</span></div>
              <div className="flex items-center gap-1"><div className="w-2 h-2 rounded-full bg-forecast" /><span>AI Predicted</span></div>
            </div>
          }
        />
        <div className="p-6">
          <div className="relative border-l-2 border-border-emphasis ml-4 space-y-8 pb-4">
            {events.map((ev: any, i: number) => {
              const isObserved = ev.observed;
              const color = isObserved ? 'text-safe' : 'text-forecast';
              const bg = isObserved ? 'bg-safe' : 'bg-forecast';
              
              return (
                <div key={ev.id || i} className="relative pl-6">
                  {/* Timeline dot */}
                  <div className={cn('absolute -left-[9px] top-1 w-4 h-4 rounded-full flex items-center justify-center border-2 border-surface-2', bg)}>
                    {isObserved ? <Check size={10} className="text-surface-0" /> : <Zap size={10} className="text-surface-0" />}
                  </div>

                  <div className="flex flex-col sm:flex-row sm:items-start gap-4">
                    {/* Time & Stage */}
                    <div className="sm:w-32 flex-shrink-0 pt-0.5">
                      <div className="font-mono text-sm font-semibold text-text-primary mb-1">
                        {isObserved ? 'Now' : `t+${ev.id}`}
                      </div>
                      <Badge severity={isObserved ? (ev.stage === 'Benign' ? 'Safe' : 'High') : 'Forecast'} className="text-[9px]">
                        {ev.stage}
                      </Badge>
                    </div>

                    {/* Content */}
                    <div className={cn('flex-1 rounded-lg border p-4', isObserved ? 'bg-surface-3 border-border-subtle' : 'bg-forecast-bg/20 border-forecast-dim')}>
                      <div className="flex items-center justify-between mb-2">
                        <h4 className={cn('text-sm font-semibold', color)}>{ev.label}</h4>
                        {!isObserved && <Badge severity="Forecast" dot className="text-[9px]">Model Estimate</Badge>}
                      </div>
                      <p className="text-xs text-text-secondary leading-relaxed">{ev.detail}</p>
                      
                      {ev.riskScore && (
                        <div className="mt-3 pt-3 border-t border-border-subtle/50 flex items-center justify-between text-[10px]">
                          <span className="text-text-muted">Associated Risk Score</span>
                          <span className={cn('font-semibold', ev.riskScore >= 80 ? 'text-critical' : ev.riskScore >= 65 ? 'text-warning' : 'text-text-primary')}>
                            {Math.round(ev.riskScore)}%
                          </span>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </Card>
    </AppShell>
  );
}
