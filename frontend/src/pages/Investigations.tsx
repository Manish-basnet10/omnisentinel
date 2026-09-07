import React, { useState, useEffect } from 'react';
import { AppShell } from '../components/layout/AppShell';
import { Card, CardHeader } from '../components/common/Card';
import { Badge, RiskBadge } from '../components/common/Badge';
import { ErrorState } from '../components/common/States';
import { useAnalysis } from '../hooks/useAnalysis';
import { Brain, Shield, AlertTriangle, FileText, ArrowRight, BookOpen } from 'lucide-react';
import { cn } from '../utils/cn';

export default function Investigations() {
  const { analysis } = useAnalysis();

  if (!analysis) {
    return (
      <AppShell>
        <div className="flex flex-col items-center justify-center h-full min-h-[60vh] text-center">
          <BookOpen size={48} className="text-text-disabled mb-4" />
          <h2 className="text-xl font-bold text-text-primary mb-2">No Dataset Uploaded</h2>
          <p className="text-text-muted">Upload a dataset to generate a contextual investigation.</p>
        </div>
      </AppShell>
    );
  }

  const invs = analysis.investigations || [];
  const inv = invs[0];

  if (!inv) {
    return (
      <AppShell>
        <ErrorState message="No investigations were generated for this dataset (risk is likely low)." />
      </AppShell>
    );
  }

  return (
    <AppShell>
      {/* Header */}
      <div className="flex items-start justify-between mb-6">
        <div>
          <div className="flex items-center gap-3 mb-1">
            <h2 className="text-xl font-bold text-text-primary tracking-tight">Investigation workspace</h2>
            <Badge severity="Muted" className="font-mono text-[10px]">{inv.id}</Badge>
            <Badge severity={inv.status === 'Open' ? 'High' : 'Muted'} dot>{inv.status}</Badge>
          </div>
          <p className="text-sm text-text-secondary">{inv.title}</p>
        </div>
        <div className="text-right">
          <RiskBadge score={analysis.prediction.current_risk} />
          <div className="text-[10px] text-text-muted mt-1.5">Opened: {analysis.created_at}</div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 mb-4">
        <div className="lg:col-span-2 space-y-4">
          <Card>
            <CardHeader title="Observed Behavior" icon={<EyeIcon />} />
            <div className="p-4 space-y-2">
              {inv.observations.map((obs: string, i: number) => (
                <div key={i} className="flex items-start gap-2 text-sm text-text-secondary">
                  <span className="text-safe mt-0.5">•</span>
                  <span>{obs}</span>
                </div>
              ))}
            </div>
          </Card>

          <Card glow="forecast">
            <CardHeader title="Predicted Behavior (AI Forecast)" icon={<Brain size={14} />} right={<Badge severity="Forecast" dot className="text-[9px]">Model Estimate</Badge>} />
            <div className="p-4 space-y-2 bg-forecast-bg/30">
              {inv.predictedBehavior.map((pred: string, i: number) => (
                <div key={i} className="flex items-start gap-2 text-sm text-forecast-bright font-medium">
                  <span className="text-forecast mt-0.5">◆</span>
                  <span>{pred}</span>
                </div>
              ))}
              {inv.predictedBehavior.length === 0 && (
                <div className="text-sm text-forecast-dim italic">No further malicious behavior predicted.</div>
              )}
            </div>
          </Card>

          <Card>
            <CardHeader title="Recommended Steps" icon={<Shield size={14} />} />
            <div className="p-4 space-y-2">
              {inv.recommendedSteps.map((step: string, i: number) => (
                <div key={i} className="flex items-center gap-2 p-2 rounded border border-border-subtle bg-surface-3">
                  <input type="checkbox" className="w-3.5 h-3.5 rounded border-border-default bg-surface-4 accent-forecast" />
                  <span className="text-xs text-text-primary">{step}</span>
                </div>
              ))}
            </div>
          </Card>
        </div>

        <div className="space-y-4">
          <Card>
            <CardHeader title="Affected Assets" icon={<AlertTriangle size={14} />} />
            <div className="p-4">
              <ul className="space-y-2">
                {inv.affectedAssets.map((asset: string) => (
                  <li key={asset} className="flex items-center gap-2 text-xs text-text-secondary">
                    <div className="w-1.5 h-1.5 rounded-full bg-critical" />
                    <span className="font-mono text-info">{asset}</span>
                  </li>
                ))}
              </ul>
            </div>
          </Card>

          <Card>
            <CardHeader title="Evidence Log" icon={<FileText size={14} />} />
            <div className="p-4 space-y-3">
              {inv.evidence.map((ev: any) => (
                <div key={ev.id} className="border-l-2 border-border-emphasis pl-3 pb-3 last:pb-0">
                  <div className="flex items-center gap-2 text-[10px] text-text-muted mb-0.5">
                    <span className="font-mono">{ev.time}</span>
                    <span>·</span>
                    <span className="uppercase tracking-wide">{ev.type}</span>
                  </div>
                  <div className="text-xs text-text-primary">{ev.description}</div>
                </div>
              ))}
            </div>
          </Card>
          
          <button className="btn btn-secondary w-full justify-center">
             <ArrowRight size={13} /> Close Investigation
          </button>
        </div>
      </div>
    </AppShell>
  );
}

function EyeIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z" />
      <circle cx="12" cy="12" r="3" />
    </svg>
  );
}
