import React, { useState, useEffect } from 'react';
import { AppShell } from '../components/layout/AppShell';
import { Card, CardHeader } from '../components/common/Card';
import { Badge } from '../components/common/Badge';
import { SkeletonCard, ProgressBar } from '../components/common/States';
import { getMitreStages } from '../api/forecastApi';
import type { mockMitreStages } from '../data/mockForecast';
import { Shield, Check, Zap, Info } from 'lucide-react';
import { cn } from '../utils/cn';

type MitreStage = typeof mockMitreStages[0];

export default function MitreAttack() {
  const [stages, setStages] = useState<MitreStage[]>([]);
  const [loading, setLoading] = useState(true);
  const [selected, setSelected] = useState<MitreStage | null>(null);

  useEffect(() => {
    getMitreStages().then(res => {
      setStages(res);
      setSelected(res.find(s => s.status === 'current') || res[0]);
      setLoading(false);
    });
  }, []);

  return (
    <AppShell>
      <div className="mb-4">
        <h2 className="text-xl font-bold text-text-primary tracking-tight">MITRE ATT&CK Matrix Mapping</h2>
        <p className="text-sm text-text-muted mt-1">Observed and AI-forecasted tactics within the enterprise matrix.</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Left: Matrix Grid */}
        <div className="lg:col-span-2 space-y-4">
          <Card>
            <CardHeader
              title="Kill Chain Progression"
              icon={<Shield size={14} />}
              right={
                <div className="flex items-center gap-3 text-[10px]">
                  <LegendItem color="bg-safe" label="Observed" />
                  <LegendItem color="bg-elevated" label="Current" />
                  <LegendItem color="bg-forecast" label="Forecast" />
                  <LegendItem color="bg-surface-3 border border-border-default" label="Not observed" />
                </div>
              }
            />
            <div className="p-4">
              {loading ? <SkeletonCard rows={6} /> : (
                <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                  {stages.map(stage => {
                    const isObserved = stage.status === 'observed';
                    const isCurrent = stage.status === 'current';
                    const isForecast = stage.status === 'forecast';
                    const isSelected = selected?.id === stage.id;
                    
                    return (
                      <button
                        key={stage.id}
                        onClick={() => setSelected(stage)}
                        className={cn(
                          'p-3 rounded text-left transition-all border text-xs relative overflow-hidden flex flex-col justify-between h-20',
                          isSelected ? 'ring-2 ring-offset-2 ring-offset-surface-2' : 'hover:border-border-emphasis',
                          isSelected && isObserved ? 'ring-safe' : '',
                          isSelected && isCurrent ? 'ring-elevated' : '',
                          isSelected && isForecast ? 'ring-forecast' : '',
                          isSelected && stage.status === 'not_observed' ? 'ring-border-emphasis' : '',
                          isObserved ? 'bg-safe-bg border-safe-dim' :
                          isCurrent ? 'bg-elevated-bg border-elevated-dim' :
                          isForecast ? 'bg-forecast-bg border-forecast-dim' :
                          'bg-surface-3 border-border-subtle opacity-70'
                        )}
                      >
                        <div className="flex justify-between items-start mb-1 z-10 relative">
                          <span className={cn('font-semibold leading-tight',
                            isObserved ? 'text-safe' : isCurrent ? 'text-elevated' : isForecast ? 'text-forecast' : 'text-text-muted'
                          )}>
                            {stage.tactic}
                          </span>
                          {isObserved && <Check size={12} className="text-safe" />}
                          {isCurrent && <div className="w-2 h-2 rounded-full bg-elevated animate-pulse" />}
                          {isForecast && <Zap size={12} className="text-forecast" />}
                        </div>
                        <div className="text-[10px] text-text-secondary truncate z-10 relative">
                          {stage.technique}
                        </div>
                        {isForecast && (
                          <div className="absolute right-0 bottom-0 p-1 opacity-20">
                            <Zap size={32} className="text-forecast" />
                          </div>
                        )}
                      </button>
                    );
                  })}
                </div>
              )}
            </div>
          </Card>
        </div>

        {/* Right: Details Panel */}
        <div>
          <Card className="h-full min-h-[300px]">
            <CardHeader title="Tactic Details" icon={<Info size={14} />} />
            {loading ? <div className="p-4"><SkeletonCard rows={5} /></div> : selected ? (
              <div className="p-5 space-y-5">
                <div>
                  <div className="flex items-center gap-2 mb-1">
                    <h3 className="text-lg font-bold text-text-primary">{selected.tactic}</h3>
                    {selected.status === 'observed' && <Badge severity="Safe" className="py-0">Observed</Badge>}
                    {selected.status === 'current'  && <Badge severity="High" className="py-0">Current</Badge>}
                    {selected.status === 'forecast' && <Badge severity="Forecast" className="py-0">Forecast</Badge>}
                    {selected.status === 'not_observed' && <Badge severity="Muted" className="py-0">Not observed</Badge>}
                  </div>
                  <div className="text-sm text-text-secondary font-mono">{selected.technique}</div>
                </div>

                <div className="space-y-1">
                  <div className="text-[10px] uppercase tracking-widest text-text-muted">Description</div>
                  <p className="text-xs text-text-primary leading-relaxed">{selected.description}</p>
                </div>

                {selected.status !== 'not_observed' && (
                  <div className="space-y-1">
                    <div className="text-[10px] uppercase tracking-widest text-text-muted">Evidence & Indicators</div>
                    <div className="p-3 bg-surface-3 border border-border-subtle rounded text-xs text-text-secondary font-mono">
                      {selected.evidence}
                    </div>
                  </div>
                )}

                {(selected.status === 'current' || selected.status === 'forecast') && (
                  <div className="space-y-1">
                    <div className="flex justify-between text-[10px] uppercase tracking-widest text-text-muted mb-1">
                      <span>Model Confidence</span>
                      <span>{selected.confidence}%</span>
                    </div>
                    <ProgressBar value={selected.confidence} color={selected.status === 'current' ? 'warning' : 'forecast'} />
                  </div>
                )}
              </div>
            ) : (
              <div className="p-4"><p className="text-sm text-text-muted">Select a tactic to view details.</p></div>
            )}
          </Card>
        </div>
      </div>
    </AppShell>
  );
}

function LegendItem({ color, label }: { color: string; label: string }) {
  return (
    <div className="flex items-center gap-1.5">
      <div className={cn('w-3 h-3 rounded', color)} />
      <span className="text-text-muted">{label}</span>
    </div>
  );
}
