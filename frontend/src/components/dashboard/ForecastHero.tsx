import React, { useState, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { Zap, ArrowRight, CheckCircle, Loader, ExternalLink } from 'lucide-react';
import { Card, CardHeader } from '../common/Card';
import { Badge } from '../common/Badge';
import { runForecast } from '../../api/forecastApi';
import type { mockForecastHero } from '../../data/mockDashboard';

type ForecastHero = typeof mockForecastHero;

interface ForecastHeroProps { forecast: ForecastHero; onRefresh?: (f: ForecastHero) => void; }

const PIPELINE_STEPS = [
  'Collecting network state',
  'Processing temporal sequence',
  'Running model',
  'Generating forecast',
  'Calculating confidence',
  'Generating explanation',
];

export function ForecastHeroCard({ forecast, onRefresh }: ForecastHeroProps) {
  const navigate = useNavigate();
  const [running, setRunning] = useState(false);
  const [currentStep, setCurrentStep] = useState(-1);
  const [completed, setCompleted] = useState(false);

  const handleRun = useCallback(async () => {
    if (running) return;
    setRunning(true);
    setCompleted(false);
    setCurrentStep(0);

    const { steps } = await runForecast(5);
    for (let i = 0; i < steps.length; i++) {
      setCurrentStep(i);
      await new Promise(r => setTimeout(r, steps[i].duration));
    }

    setRunning(false);
    setCompleted(true);
    setCurrentStep(-1);
    onRefresh?.({
      ...forecast,
      progressionProbability: Math.min(98, forecast.progressionProbability + Math.floor(Math.random() * 4 - 1)),
      timestamp: new Date().toISOString(),
    });
  }, [running, forecast, onRefresh]);

  return (
    <Card glow="forecast" className="col-span-full">
      <CardHeader
        title="AI Attack Forecast"
        subtitle="What is likely to happen next?"
        icon={<Zap size={14} />}
        right={
          <div className="flex items-center gap-2">
            <Badge severity="Forecast" dot>MODEL ESTIMATE</Badge>
            <span className="text-[10px] text-text-muted font-mono">
              {new Date(forecast.timestamp).toLocaleTimeString('en-GB', { hour12: false })}
            </span>
          </div>
        }
      />

      <div className="p-4 grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Stage progression */}
        <div className="lg:col-span-2 space-y-4">
          {/* Three stages */}
          <div className="grid grid-cols-3 gap-3">
            <StageBox
              label="Current Stage"
              stage={forecast.currentStage}
              variant="current"
              description="Observed behavior"
            />
            <StageBox
              label="Predicted Next Stage"
              stage={forecast.predictedNextStage}
              variant="forecast"
              description={`${forecast.progressionProbability}% estimated probability`}
              highlight
            />
            <StageBox
              label="Possible Following"
              stage={forecast.possibleFollowingStage}
              variant="dimForecast"
              description="Contingent forecast"
            />
          </div>

          {/* Pipeline status */}
          <AnimatePresence>
            {running && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
                className="space-y-1.5 overflow-hidden"
              >
                {PIPELINE_STEPS.map((step, i) => (
                  <PipelineStep
                    key={step}
                    label={step}
                    state={i < currentStep ? 'done' : i === currentStep ? 'running' : 'pending'}
                    index={i + 1}
                  />
                ))}
              </motion.div>
            )}
          </AnimatePresence>

          {completed && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="flex items-center gap-2 text-safe text-xs"
            >
              <CheckCircle size={13} />
              <span>Forecast updated · {new Date().toLocaleTimeString('en-GB', { hour12: false })}</span>
            </motion.div>
          )}
        </div>

        {/* Right: metrics + actions */}
        <div className="space-y-4 flex flex-col justify-between">
          <div className="space-y-3">
            <MetricRow label="Probability of Progression" value={`${forecast.progressionProbability}%`} color="text-warning" />
            <MetricRow label="Confidence Level"           value={forecast.confidenceLevel}             color="text-forecast" />
            <MetricRow label="Forecast Horizon"           value={`Next ${forecast.forecastHorizon} windows`} color="text-text-secondary" />
          </div>

          <div className="space-y-2">
            <button
              onClick={handleRun}
              disabled={running}
              className="btn btn-primary w-full justify-center"
              id="run-forecast-btn"
            >
              {running ? <Loader size={13} className="animate-spin" /> : <Zap size={13} />}
              {running ? 'Running…' : 'Run Forecast'}
            </button>
            <button
              onClick={() => navigate('/attack-forecast')}
              className="btn btn-secondary w-full justify-center"
            >
              <ExternalLink size={13} />
              View Full Forecast
            </button>
          </div>
        </div>
      </div>
    </Card>
  );
}

function StageBox({ label, stage, variant, description, highlight }: {
  label: string; stage: string; variant: 'current' | 'forecast' | 'dimForecast';
  description?: string; highlight?: boolean;
}) {
  const borderColor =
    variant === 'current'     ? 'border-elevated-dim bg-elevated-bg' :
    variant === 'forecast'    ? 'border-forecast bg-forecast-bg border-glow-forecast' :
                                'border-border-emphasis bg-surface-3';
  const stageColor =
    variant === 'current'     ? 'text-elevated' :
    variant === 'forecast'    ? 'text-forecast-bright' :
                                'text-text-secondary';

  return (
    <div className={`rounded border p-3 space-y-1.5 ${borderColor} ${highlight ? 'ring-1 ring-forecast/30' : ''}`}>
      <div className="text-[10px] uppercase tracking-widest text-text-muted">{label}</div>
      <div className={`text-sm font-bold ${stageColor}`}>{stage}</div>
      {description && <div className="text-[10px] text-text-muted">{description}</div>}
    </div>
  );
}

function MetricRow({ label, value, color }: { label: string; value: string; color: string }) {
  return (
    <div className="flex items-center justify-between border-b border-border-subtle pb-2">
      <span className="text-[11px] text-text-muted">{label}</span>
      <span className={`text-xs font-semibold ${color}`}>{value}</span>
    </div>
  );
}

function PipelineStep({ label, state, index }: { label: string; state: 'pending' | 'running' | 'done'; index: number }) {
  return (
    <div className="flex items-center gap-2 text-[11px]">
      <div className="w-4 h-4 flex items-center justify-center flex-shrink-0">
        {state === 'done'    && <CheckCircle size={13} className="text-safe" />}
        {state === 'running' && <Loader size={13} className="text-forecast animate-spin" />}
        {state === 'pending' && <span className="w-2 h-2 rounded-full bg-border-emphasis" />}
      </div>
      <span className={state === 'done' ? 'text-text-muted' : state === 'running' ? 'text-text-primary' : 'text-text-disabled'}>
        {index}. {label}
      </span>
    </div>
  );
}
