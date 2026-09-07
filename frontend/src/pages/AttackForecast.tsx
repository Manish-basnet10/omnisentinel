import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { AppShell } from '../components/layout/AppShell';
import { Card, CardHeader } from '../components/common/Card';
import { Badge } from '../components/common/Badge';
import { ProgressBar, Spinner } from '../components/common/States';
import { useAnalysis } from '../hooks/useAnalysis';
import { Zap, Play, ArrowRight, BarChart2 } from 'lucide-react';
import { cn } from '../utils/cn';
import { confidenceLabel } from '../utils/formatters';

const STAGE_COLOR: Record<string, string> = {
  Reconnaissance: '#f97316',
  Scanning: '#f97316',
  'Initial Access': '#ef4444',
  'Lateral Movement': '#ef4444',
  'Command & Control': '#dc2626',
  Exfiltration: '#dc2626',
  Impact: '#b91c1c',
  Benign: '#22c55e',
};

// Map backend forecast to UI shape
function transformForecast(analysis: any, horizon: number) {
  if (!analysis) return { steps: [], overallRisk: 0 };
  
  const currentRisk = analysis.prediction.current_risk;
  const currentStage = analysis.mitre_progression[0]?.tactic || 'Unknown';
  
  const steps = [];
  
  // Now step
  steps.push({
    horizon: 'Now',
    stage: currentStage,
    riskScore: Math.round(currentRisk),
    probability: 100, // already observed
    confidence: Math.round(analysis.prediction.confidence || 0),
    description: 'Current network state.',
    trafficPattern: 'Observed traffic pattern',
  });

  // Future steps
  const backendForecast = analysis.forecast || [];
  const limit = Math.min(horizon, backendForecast.length);
  
  for (let i = 0; i < limit; i++) {
    const f = backendForecast[i];
    steps.push({
      horizon: `+${i + 1}`,
      stage: f.mitre_stage || analysis.prediction.predicted_next_stage || 'Unknown',
      riskScore: Math.round(f.risk_score || currentRisk),
      probability: Math.round(f.probability ? f.probability * 100 : analysis.prediction.attack_probability * 100),
      confidence: Math.round(f.confidence ? f.confidence * 100 : analysis.prediction.confidence),
      description: `Model prediction for t+${i+1}`,
      trafficPattern: 'Predicted pattern',
    });
  }
  
  const lastStepRisk = steps.length > 1 ? steps[steps.length - 1].riskScore : currentRisk;

  return {
    steps,
    overallRisk: lastStepRisk,
    runAt: analysis.created_at,
  };
}

export default function AttackForecast() {
  const { analysis } = useAnalysis();
  const [horizon, setHorizon] = useState<number>(5);
  const [simulating, setSimulating] = useState(false);
  const [simStep, setSimStep] = useState(-1);

  if (!analysis) {
    return (
      <AppShell>
        <div className="flex flex-col items-center justify-center h-full min-h-[60vh] text-center">
          <BarChart2 size={48} className="text-text-disabled mb-4" />
          <h2 className="text-xl font-bold text-text-primary mb-2">No Dataset Uploaded</h2>
          <p className="text-text-muted">Upload a dataset to generate an attack forecast.</p>
        </div>
      </AppShell>
    );
  }

  const forecast = transformForecast(analysis, horizon);
  const steps = forecast.steps;

  const handleSimulate = async () => {
    if (simulating) return;
    setSimulating(true);
    for (let i = 0; i < steps.length; i++) {
      setSimStep(i);
      await new Promise(r => setTimeout(r, 700));
    }
    setSimulating(false);
    setSimStep(-1);
  };

  return (
    <AppShell>
      {/* Page header */}
      <div className="mb-6">
        <div className="flex items-start justify-between">
          <div>
            <h2 className="text-xl font-bold text-text-primary tracking-tight">ATTACK FORECAST ENGINE</h2>
            <p className="text-sm text-text-muted mt-1">Predicting where the current network trajectory is heading based on the active dataset.</p>
            <Badge severity="Forecast" dot className="mt-2">Live Model Output</Badge>
          </div>
          <div className="flex items-center gap-2">
            {/* Horizon selector */}
            <div className="flex items-center gap-1 bg-surface-2 border border-border-subtle rounded p-1">
              {[1, 3, 5, 8].map(h => (
                <button
                  key={h}
                  onClick={() => setHorizon(h)}
                  className={cn(
                    'px-3 py-1 rounded text-xs font-medium transition-colors',
                    horizon === h ? 'bg-forecast-dim text-forecast-bright' : 'text-text-muted hover:text-text-primary'
                  )}
                >
                  +{h}w
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Current + Forecast metrics */}
      <div className="grid grid-cols-3 gap-3 mb-4">
        <div className="card p-4">
          <div className="text-[10px] text-text-muted uppercase tracking-widest mb-1">Current Risk</div>
          <div className="text-3xl font-bold text-warning">{Math.round(analysis.prediction.current_risk)}%</div>
          <ProgressBar value={analysis.prediction.current_risk} color="warning" className="mt-2" />
        </div>
        <div className="card p-4 border-glow-forecast">
          <div className="text-[10px] text-forecast uppercase tracking-widest mb-1">Forecast Risk (+{horizon})</div>
          <div className="text-3xl font-bold text-forecast-bright">{forecast.overallRisk}%</div>
          <ProgressBar value={forecast.overallRisk} color="forecast" className="mt-2" />
        </div>
        <div className="card p-4">
          <div className="text-[10px] text-text-muted uppercase tracking-widest mb-1">Confidence</div>
          <div className="text-3xl font-bold text-text-primary">
            {steps[0]?.confidence ?? 0}%
          </div>
          <div className="text-[11px] text-text-muted mt-1">{confidenceLabel(steps[0]?.confidence ?? 0)}</div>
        </div>
      </div>

      {/* Forecast Trajectory */}
      <Card className="mb-4">
        <CardHeader
          title="Forecast Trajectory"
          subtitle={`Horizon: +${horizon} observation windows`}
          icon={<Zap size={14} />}
          right={
            <span className="text-[10px] text-text-muted font-mono">
              Model: {analysis.model.type}
            </span>
          }
        />
        <div className="p-4 overflow-x-auto">
          <div className="flex items-start gap-0 min-w-max">
            {steps.map((step: any, i: number) => (
              <React.Fragment key={step.horizon}>
                <StageCard step={step} isNow={step.horizon === 'Now'} />
                {i < steps.length - 1 && (
                  <div className="flex items-center mt-8 flex-shrink-0">
                    <div className={cn('h-0.5 w-8', step.horizon === 'Now' ? 'bg-warning' : 'bg-forecast opacity-40')} />
                    <ArrowRight size={12} className={cn(step.horizon === 'Now' ? 'text-warning' : 'text-forecast opacity-40')} />
                  </div>
                )}
              </React.Fragment>
            ))}
          </div>
        </div>
      </Card>

      {/* Forward Simulation */}
      <Card>
        <CardHeader
          title="Forward Simulation"
          subtitle="S(t) → S(t+1) → S(t+2) → … Autoregressive state progression"
          icon={<Play size={14} />}
          right={
            <button onClick={handleSimulate} disabled={simulating} className="btn btn-secondary btn-sm">
              {simulating ? <Spinner size={12} /> : <Play size={12} />}
              {simulating ? 'Simulating…' : 'Simulate Future'}
            </button>
          }
        />
        <div className="p-4 overflow-x-auto">
          <div className="flex items-stretch gap-3 min-w-max">
            {steps.map((step: any, i: number) => {
              const active = simulating && i <= simStep;
              return (
                <motion.div
                  key={step.horizon}
                  animate={active ? { scale: 1.02, opacity: 1 } : { scale: 1, opacity: simulating && i > simStep ? 0.3 : 1 }}
                  transition={{ duration: 0.3 }}
                  className={cn(
                    'w-40 rounded border p-3 flex flex-col gap-2',
                    active ? 'border-forecast bg-forecast-bg' : 'border-border-subtle bg-surface-2'
                  )}
                >
                  <div className="text-[10px] font-semibold text-text-muted">S{step.horizon === 'Now' ? '(t)' : `(t${step.horizon})`}</div>
                  <div className="text-xs font-bold" style={{ color: STAGE_COLOR[step.stage] || '#94a3b8' }}>
                    {step.stage}
                  </div>
                  <div className="space-y-1">
                    <div className="flex justify-between text-[10px]">
                      <span className="text-text-muted">Risk</span>
                      <span className="text-text-secondary">{step.riskScore}%</span>
                    </div>
                    <ProgressBar value={step.riskScore} color={step.riskScore >= 80 ? 'critical' : step.riskScore >= 65 ? 'warning' : 'forecast'} />
                  </div>
                  {step.horizon !== 'Now' && (
                    <div className="text-[10px] text-forecast">
                      {step.probability}% est. probability
                    </div>
                  )}
                </motion.div>
              );
            })}
          </div>
        </div>
      </Card>
    </AppShell>
  );
}

function StageCard({ step, isNow }: { step: any; isNow: boolean }) {
  return (
    <div className={cn(
      'w-44 rounded border p-3 space-y-2',
      isNow ? 'border-elevated bg-elevated-bg' : 'border-forecast-dim bg-forecast-bg'
    )}>
      <div className="flex items-center justify-between">
        <span className="text-[10px] font-mono text-text-muted">{step.horizon === 'Now' ? 'NOW' : step.horizon}</span>
        {isNow
          ? <Badge severity="Medium" className="text-[9px] py-0">Current</Badge>
          : <Badge severity="Forecast" className="text-[9px] py-0">Forecast</Badge>
        }
      </div>
      <div className="text-sm font-bold" style={{ color: STAGE_COLOR[step.stage] || '#a78bfa' }}>
        {step.stage}
      </div>
      <div className="space-y-1">
        <div className="flex justify-between text-[10px]">
          <span className="text-text-muted">Probability</span>
          <span className="font-semibold text-text-secondary">{step.probability}%</span>
        </div>
        <ProgressBar value={step.probability} color={isNow ? 'warning' : 'forecast'} />
        <div className="flex justify-between text-[10px]">
          <span className="text-text-muted">Confidence</span>
          <span className="text-text-secondary">{step.confidence}%</span>
        </div>
      </div>
      <p className="text-[10px] text-text-muted leading-relaxed">{step.description}</p>
    </div>
  );
}
