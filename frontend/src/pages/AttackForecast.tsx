import React, { useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { AppShell } from '../components/layout/AppShell';
import { Card, CardHeader } from '../components/common/Card';
import { Badge } from '../components/common/Badge';
import { ProgressBar, Spinner } from '../components/common/States';
import { getLatestForecast, runForecast } from '../api/forecastApi';
import { mockForecastSteps } from '../data/mockForecast';
import { Zap, CheckCircle, Loader, Play, ArrowRight, FlaskConical } from 'lucide-react';
import { cn } from '../utils/cn';
import { confidenceLabel } from '../utils/formatters';

type Horizon = 1 | 3 | 5 | 10;
const HORIZONS: Horizon[] = [1, 3, 5, 10];

const PIPELINE_STEPS = [
  'Collecting network state',
  'Processing temporal sequence',
  'Running model',
  'Generating forecast',
  'Calculating confidence',
  'Generating explanation',
];

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

export default function AttackForecast() {
  const [horizon, setHorizon] = useState<Horizon>(5);
  const [running, setRunning] = useState(false);
  const [pipelineStep, setPipelineStep] = useState(-1);
  const [forecast, setForecast] = useState<any>(() => ({
    steps: mockForecastSteps[5],
    horizon: 5,
    overallRisk: 78,
    runAt: new Date().toISOString(),
  }));
  const [simulating, setSimulating] = useState(false);
  const [simStep, setSimStep] = useState(-1);

  const handleRunForecast = useCallback(async () => {
    if (running) return;
    setRunning(true);
    setPipelineStep(0);
    const { steps } = await runForecast(horizon);
    for (let i = 0; i < steps.length; i++) {
      setPipelineStep(i);
      await new Promise(r => setTimeout(r, steps[i].duration));
    }
    const result = await getLatestForecast(horizon);
    setForecast(result);
    setRunning(false);
    setPipelineStep(-1);
  }, [horizon, running]);

  const handleSimulate = useCallback(async () => {
    if (simulating) return;
    setSimulating(true);
    const steps = forecast.steps || [];
    for (let i = 0; i < steps.length; i++) {
      setSimStep(i);
      await new Promise(r => setTimeout(r, 700));
    }
    setSimulating(false);
    setSimStep(-1);
  }, [simulating, forecast.steps]);

  const steps = forecast?.steps || [];

  return (
    <AppShell>
      {/* Page header */}
      <div className="mb-6">
        <div className="flex items-start justify-between">
          <div>
            <h2 className="text-xl font-bold text-text-primary tracking-tight">ATTACK FORECAST ENGINE</h2>
            <p className="text-sm text-text-muted mt-1">Predicting where the current network trajectory is heading.</p>
            <Badge severity="Forecast" dot className="mt-2">Model Estimate · Demo Data</Badge>
          </div>
          <div className="flex items-center gap-2">
            {/* Horizon selector */}
            <div className="flex items-center gap-1 bg-surface-2 border border-border-subtle rounded p-1">
              {HORIZONS.map(h => (
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
            <button
              onClick={handleRunForecast}
              disabled={running}
              className="btn btn-primary"
              id="run-forecast-full-btn"
            >
              {running ? <Loader size={13} className="animate-spin" /> : <Zap size={13} />}
              {running ? 'Running…' : 'Run Forecast'}
            </button>
          </div>
        </div>
      </div>

      {/* Current + Forecast metrics */}
      <div className="grid grid-cols-3 gap-3 mb-4">
        <div className="card p-4">
          <div className="text-[10px] text-text-muted uppercase tracking-widest mb-1">Current Risk</div>
          <div className="text-3xl font-bold text-warning">72%</div>
          <ProgressBar value={72} color="warning" className="mt-2" />
        </div>
        <div className="card p-4 border-glow-forecast">
          <div className="text-[10px] text-forecast uppercase tracking-widest mb-1">Forecast Risk</div>
          <div className="text-3xl font-bold text-forecast-bright">{forecast.overallRisk}%</div>
          <ProgressBar value={forecast.overallRisk} color="forecast" className="mt-2" />
        </div>
        <div className="card p-4">
          <div className="text-[10px] text-text-muted uppercase tracking-widest mb-1">Confidence</div>
          <div className="text-3xl font-bold text-text-primary">
            {steps[0]?.confidence ?? 88}%
          </div>
          <div className="text-[11px] text-text-muted mt-1">{confidenceLabel(steps[0]?.confidence ?? 88)}</div>
        </div>
      </div>

      {/* Pipeline status */}
      <AnimatePresence>
        {running && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="card p-4 mb-4 overflow-hidden"
          >
            <div className="text-xs font-semibold text-text-secondary mb-3">Running inference pipeline…</div>
            <div className="grid grid-cols-3 gap-2">
              {PIPELINE_STEPS.map((step, i) => (
                <div key={step} className={cn(
                  'flex items-center gap-2 px-3 py-2 rounded text-[11px] border',
                  i < pipelineStep ? 'bg-safe-bg border-safe-dim text-safe' :
                  i === pipelineStep ? 'bg-forecast-bg border-forecast text-forecast-bright' :
                  'bg-surface-2 border-border-subtle text-text-disabled'
                )}>
                  {i < pipelineStep  && <CheckCircle size={11} />}
                  {i === pipelineStep && <Loader size={11} className="animate-spin" />}
                  {i > pipelineStep  && <span className="w-3 h-3 rounded-full border border-current opacity-30" />}
                  {step}
                </div>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Forecast Trajectory */}
      <Card className="mb-4">
        <CardHeader
          title="Forecast Trajectory"
          subtitle={`Horizon: +${horizon} observation windows`}
          icon={<Zap size={14} />}
          right={
            <span className="text-[10px] text-text-muted font-mono">
              Updated: {new Date(forecast.runAt).toLocaleTimeString('en-GB', { hour12: false })}
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
              {simulating ? <Loader size={12} className="animate-spin" /> : <Play size={12} />}
              {simulating ? 'Simulating…' : 'Simulate Future'}
            </button>
          }
        />
        <div className="p-4 overflow-x-auto">
          <div className="flex items-stretch gap-3 min-w-max">
            {steps.map((step: any, i: number) => {
              const active = simulating && i <= simStep;
              const past = !simulating && simStep === -1 && i === 0;
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
                    <div className="text-[10px] text-text-muted">{step.trafficPattern}</div>
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
          <p className="text-[10px] text-text-muted mt-3 flex items-center gap-1">
            <FlaskConical size={10} />
            Demo Data — Simulated autoregressive rollout. Not real model output.
          </p>
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
