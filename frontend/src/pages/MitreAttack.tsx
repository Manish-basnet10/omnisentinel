import React, { useState, useEffect } from 'react';
import { AppShell } from '../components/layout/AppShell';
import { Card, CardHeader } from '../components/common/Card';
import { Badge } from '../components/common/Badge';
import { ProgressBar } from '../components/common/States';
import { useAnalysis } from '../hooks/useAnalysis';
import { Shield, Check, Zap, Info, Database } from 'lucide-react';
import { cn } from '../utils/cn';

// All standard MITRE Tactics we want to show on the board
const STANDARD_TACTICS = [
  'Reconnaissance', 'Resource Development', 'Initial Access',
  'Execution', 'Persistence', 'Privilege Escalation',
  'Defense Evasion', 'Credential Access', 'Discovery',
  'Lateral Movement', 'Collection', 'Command & Control',
  'Exfiltration', 'Impact'
];

interface MatrixNode {
  id: string;
  tactic: string;
  technique: string;
  description: string;
  evidence: string;
  status: 'observed' | 'current' | 'forecast' | 'not_observed';
  confidence: number;
}

function buildMatrixNodes(analysis: any): MatrixNode[] {
  if (!analysis) return [];

  const progression = analysis.mitre_progression || [];
  
  // Track what we've mapped so we know what is "not_observed"
  const mappedTactics = new Set<string>();
  const nodes: MatrixNode[] = [];

  // Map active progression from PyTorch output
  progression.forEach((m: any, idx: number) => {
    mappedTactics.add(m.tactic);
    nodes.push({
      id: `node-${idx}`,
      tactic: m.tactic,
      technique: m.technique,
      description: m.description || `Predicted activity corresponding to ${m.technique}`,
      evidence: `Confidence: ${Math.round(m.probability * 100)}% based on temporal pattern`,
      status: m.status as any, // 'current' or 'forecast'
      confidence: Math.round(m.probability * 100)
    });
  });

  // Fill in the rest of the matrix with "not_observed"
  STANDARD_TACTICS.forEach(tactic => {
    if (!mappedTactics.has(tactic)) {
      nodes.push({
        id: `node-${tactic.toLowerCase().replace(' ', '-')}`,
        tactic,
        technique: '—',
        description: 'No activity detected for this tactic in the current dataset.',
        evidence: 'N/A',
        status: 'not_observed',
        confidence: 0
      });
    }
  });

  // Sort logically (active ones first, or you could sort by kill-chain order)
  return nodes.sort((a, b) => {
    const aOrder = a.status !== 'not_observed' ? 0 : 1;
    const bOrder = b.status !== 'not_observed' ? 0 : 1;
    return aOrder - bOrder;
  });
}

export default function MitreAttack() {
  const { analysis } = useAnalysis();
  const [nodes, setNodes] = useState<MatrixNode[]>([]);
  const [selected, setSelected] = useState<MatrixNode | null>(null);

  useEffect(() => {
    if (analysis) {
      const built = buildMatrixNodes(analysis);
      setNodes(built);
      setSelected(built.find(n => n.status === 'current') || built[0]);
    } else {
      setNodes([]);
      setSelected(null);
    }
  }, [analysis]);

  if (!analysis) {
    return (
      <AppShell>
        <div className="flex flex-col items-center justify-center h-full min-h-[60vh] text-center">
          <Shield size={48} className="text-text-disabled mb-4" />
          <h2 className="text-xl font-bold text-text-primary mb-2">No Dataset Uploaded</h2>
          <p className="text-text-muted">Upload a dataset to view MITRE ATT&CK mappings.</p>
        </div>
      </AppShell>
    );
  }

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
                  <LegendItem color="bg-elevated" label="Current" />
                  <LegendItem color="bg-forecast" label="Forecast" />
                  <LegendItem color="bg-surface-3 border border-border-default" label="Not observed" />
                </div>
              }
            />
            <div className="p-4">
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                {nodes.map(node => {
                  const isCurrent = node.status === 'current';
                  const isForecast = node.status === 'forecast';
                  const isSelected = selected?.id === node.id;
                  
                  return (
                    <button
                      key={node.id}
                      onClick={() => setSelected(node)}
                      className={cn(
                        'p-3 rounded text-left transition-all border text-xs relative overflow-hidden flex flex-col justify-between h-20',
                        isSelected ? 'ring-2 ring-offset-2 ring-offset-surface-2' : 'hover:border-border-emphasis',
                        isSelected && isCurrent ? 'ring-elevated' : '',
                        isSelected && isForecast ? 'ring-forecast' : '',
                        isSelected && node.status === 'not_observed' ? 'ring-border-emphasis' : '',
                        isCurrent ? 'bg-elevated-bg border-elevated-dim' :
                        isForecast ? 'bg-forecast-bg border-forecast-dim' :
                        'bg-surface-3 border-border-subtle opacity-70'
                      )}
                    >
                      <div className="flex justify-between items-start mb-1 z-10 relative">
                        <span className={cn('font-semibold leading-tight',
                          isCurrent ? 'text-elevated' : isForecast ? 'text-forecast' : 'text-text-muted'
                        )}>
                          {node.tactic}
                        </span>
                        {isCurrent && <div className="w-2 h-2 rounded-full bg-elevated animate-pulse" />}
                        {isForecast && <Zap size={12} className="text-forecast" />}
                      </div>
                      <div className="text-[10px] text-text-secondary truncate z-10 relative">
                        {node.technique}
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
            </div>
          </Card>
        </div>

        {/* Right: Details Panel */}
        <div>
          <Card className="h-full min-h-[300px]">
            <CardHeader title="Tactic Details" icon={<Info size={14} />} />
            {selected ? (
              <div className="p-5 space-y-5">
                <div>
                  <div className="flex items-center gap-2 mb-1">
                    <h3 className="text-lg font-bold text-text-primary">{selected.tactic}</h3>
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
