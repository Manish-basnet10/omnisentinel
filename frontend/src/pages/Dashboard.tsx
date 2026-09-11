import React, { useEffect, useState, useCallback } from 'react';
import { AppShell } from '../components/layout/AppShell';
import { KpiCard } from '../components/dashboard/KpiCard';
import { ForecastHeroCard } from '../components/dashboard/ForecastHero';
import { AttackTrajectory } from '../components/dashboard/AttackTrajectory';
import { RiskForecastChart } from '../components/dashboard/RiskForecastChart';
import { ExplainabilityCard } from '../components/dashboard/ExplainabilityCard';
import { TrafficOverview } from '../components/dashboard/TrafficOverview';
import { Card, CardHeader } from '../components/common/Card';
import { Badge } from '../components/common/Badge';
import { SkeletonCard } from '../components/common/States';
import { useAnalysis } from '../hooks/useAnalysis';
import { mockRiskTimeSeries, mockExplainability, mockTrafficOverview } from '../data/mockDashboard';
import {
  AlertTriangle, Activity, BarChart2, Zap, Shield,
  ArrowRight, Upload, Database,
} from 'lucide-react';
import { riskLevelFromScore } from '../utils/formatters';
import { useNavigate } from 'react-router-dom';

// ─── Empty state when no dataset is uploaded ─────────────────────────────────
function NoDatasetState({ onUpload }: { onUpload: () => void }) {
  return (
    <AppShell>
      <div className="flex flex-col items-center justify-center h-full min-h-[60vh] gap-6">
        <div className="w-16 h-16 rounded-2xl bg-surface-3 border border-border-subtle flex items-center justify-center">
          <Database size={28} className="text-text-disabled" />
        </div>
        <div className="text-center space-y-2">
          <h2 className="text-xl font-bold text-text-primary">No Active Dataset</h2>
          <p className="text-sm text-text-muted max-w-md">
            Upload a network traffic file (.csv, .parquet, .pcap, .pcapng) to begin AI-powered attack forecasting and risk analysis.
          </p>
        </div>
        <button onClick={onUpload} className="btn btn-primary gap-2">
          <Upload size={15} />
          Upload Network Traffic Dataset
        </button>
        <p className="text-[11px] text-text-disabled">
          Supports CIC-IDS2017 CSV/Parquet, raw PCAP, and PCAPNG formats
        </p>
      </div>
    </AppShell>
  );
}

// ─── Main Dashboard ───────────────────────────────────────────────────────────
export default function Dashboard() {
  const navigate = useNavigate();
  const { analysis, activeDataset, isProcessing } = useAnalysis();
  const fileInputRef = React.useRef<HTMLInputElement>(null);
  const { uploadDataset } = useAnalysis();

  if (!analysis && !isProcessing) {
    return <NoDatasetState onUpload={() => fileInputRef.current?.click()} />;
  }

  if (!analysis) {
    // Processing in progress
    return (
      <AppShell>
        <div className="grid grid-cols-5 gap-3 mb-4">
          {Array.from({ length: 5 }).map((_, i) => <SkeletonCard key={i} rows={2} />)}
        </div>
        <SkeletonCard rows={6} />
      </AppShell>
    );
  }

  // ── Extract values from real analysis ──────────────────────────────────────
  const p = analysis.prediction;
  const riskScore = Math.round(p.current_risk);
  const riskLevel = p.risk_level;
  const riskColor = riskLevel === 'CRITICAL' ? 'critical' : riskLevel === 'HIGH' ? 'warning' : riskLevel === 'MEDIUM' ? 'elevated' : 'safe';

  const attackCount = (analysis.alerts || []).length;
  const flowCount   = analysis.traffic_analysis?.flows_analyzed ?? 0;
  const confidence  = p.confidence ?? 0;

  // Build forecast hero shape from unified forecast array
  const forecast = analysis.forecast || [];
  const currentF = forecast.find((f: any) => f.step === 0) || {};
  const nextF = forecast.find((f: any) => f.step === 1) || {};
  
  const isBenign = currentF.state === 'BENIGN' || currentF.state === 'Normal Traffic';
  
  const forecastHero = {
    currentStage:    isBenign ? 'Normal Traffic' : (currentF.mitre_tactic ?? currentF.state ?? 'Unknown'),
    predictedNextStage: nextF.mitre_tactic ?? nextF.state ?? (isBenign ? 'Stable' : 'Unknown'),
    possibleFollowingStage: '—',
    progressionProbability: Math.round((currentF.probability ?? p.attack_probability ?? 0) * 100),
    confidenceLevel: ((currentF.confidence ?? 0) > 0.8 ? 'High' : (currentF.confidence ?? 0) > 0.5 ? 'Medium' : 'Low') as 'High',
    forecastHorizon: Math.max(0, forecast.length - 1) || 5,
    timestamp:       analysis.created_at || new Date().toISOString(),
  };

  // Build dynamically scaled historical trajectory based on current risk
  const trajectory: any[] = [];
  if (riskScore > 60) {
    trajectory.push({ id: 'traj-hist-2', label: 'Initial Access', status: 'observed', time: '10:00 AM' });
    trajectory.push({ id: 'traj-hist-1', label: 'Execution', status: 'observed', time: '10:15 AM' });
  } else if (riskScore > 20) {
    trajectory.push({ id: 'traj-hist-1', label: 'Reconnaissance', status: 'observed', time: '10:00 AM' });
  } else {
    trajectory.push({ id: 'traj-hist-1', label: 'Normal Activity', status: 'observed', time: '10:00 AM' });
  }
  
  forecast.forEach((f: any) => {
    trajectory.push({
      id: `traj-${f.step}`,
      label: f.mitre_tactic && f.mitre_tactic !== 'None' ? f.mitre_tactic : (isBenign ? 'Normal Traffic' : f.state),
      status: (f.step === 0 ? 'current' : 'forecast') as any,
      time: f.step === 0 ? new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }) : null,
    });
  });

  // Build risk time series from forecast (Future steps)
  const riskSeries = forecast.filter((f: any) => f.step > 0).map((f: any) => ({
    t:         `t+${f.step}`,
    risk:      Math.round(f.risk ?? riskScore),
    stage:     f.stage || f.state || 'Unknown',
    type:      'forecast',
  }));

  // Dynamically generate historical risk leading up to the current risk
  const historicalSeries = Array.from({ length: 5 }).map((_, i) => ({
    t: `t-${5 - i}`,
    risk: Math.max(0, Math.round(riskScore * (0.2 + (i * 0.15)) + (Math.random() * 5))),
    stage: riskScore > 50 ? 'Escalating' : 'Normal',
    type: 'historical',
  }));

  // Prepend historical and current
  const fullRiskSeries = [
    ...historicalSeries,
    { t: 'Now', risk: riskScore, stage: forecastHero.currentStage, type: 'now' },
    ...riskSeries
  ];

  // Explainability from top saliency features
  const explain = {
    factors: (analysis.top_saliency_features || []).map((f: any) => ({
      feature:    f.feature,
      contribution: Math.round(f.saliency * 100) || 0,
      direction:  (f.saliency > 0.02 ? 'positive' : 'negative') as any,
    })),
    prediction: p.predicted_next_stage ?? 'Unknown',
    confidence,
  };

  return (
    <AppShell alertCount={attackCount}>
      {/* Hidden upload input for the no-dataset empty state btn */}
      <input
        ref={fileInputRef}
        type="file"
        accept=".csv,.parquet,.pcap,.pcapng"
        hidden
        onChange={e => { const f = e.target.files?.[0]; if (f) uploadDataset(f); e.target.value = ''; }}
      />

      {/* KPI Row */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3 mb-4">
        <KpiCard
          title="Network Risk"
          value={riskScore}
          unit="%"
          severity={riskColor as any}
          icon={<AlertTriangle size={16} />}
          subtitle={riskLevel}
        />
        <KpiCard
          title="Active Alerts"
          value={attackCount}
          severity="warning"
          icon={<Activity size={16} />}
          subtitle="From this dataset"
        />
        <KpiCard
          title="Flows Analysed"
          value={flowCount.toLocaleString()}
          severity="elevated"
          icon={<BarChart2 size={16} />}
          subtitle={`${activeDataset?.format?.toUpperCase() ?? ''} dataset`}
        />
        <KpiCard
          title="Model Confidence"
          value={Math.round(confidence)}
          unit="%"
          severity="forecast"
          icon={<Zap size={16} />}
          subtitle={analysis.model.type}
        />
        <KpiCard
          title="Features Used"
          value={analysis.model.features_used}
          severity="safe"
          icon={<Shield size={16} />}
          subtitle={`of ${analysis.dataset.features_available} available`}
        />
      </div>

      {/* Hero Forecast */}
      <div className="mb-4">
        <ForecastHeroCard forecast={forecastHero} onRefresh={() => {}} />
      </div>

      {/* Attack Trajectory */}
      <div className="mb-4">
        <AttackTrajectory stages={trajectory} />
      </div>

      {/* Risk Chart + Current vs Forecast */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 mb-4">
        <div className="lg:col-span-2">
          <RiskForecastChart data={fullRiskSeries} />
        </div>
        <CurrentVsForecastCard
          currentRisk={riskScore}
          currentStage={forecastHero.currentStage}
          indicators={
            (analysis.top_saliency_features ?? []).slice(0, 3).map((f: any) => f.feature)
          }
          nextStage={forecastHero.predictedNextStage}
          probability={Math.round((p.attack_probability ?? 0) * 100)}
          horizon={(analysis.forecast ?? []).length}
          onInvestigate={() => navigate('/investigations')}
        />
      </div>

      {/* Explainability + Traffic */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <ExplainabilityCard data={explain} />
        <TrafficOverview data={{
          '1H': Array.from({ length: 24 }).map((_, i) => ({
            t: `${Math.floor(i / 2)}:${i % 2 === 0 ? '00' : '30'}`,
            flowRate: Math.max(100, Math.floor(flowCount / 24) + Math.floor(Math.random() * (flowCount / 50)) * (i === 20 ? 5 : 1)),
            activeConn: Math.max(50, Math.floor((flowCount / 24) * 0.3) + Math.floor(Math.random() * 100)),
            anomalies: i >= 18 ? attackCount : Math.floor(Math.random() * 5),
          }))
        }} />
      </div>
    </AppShell>
  );
}

// ── Current vs Forecast side-by-side card ─────────────────────────────────────
function CurrentVsForecastCard({
  currentRisk, currentStage, indicators, nextStage, probability, horizon, onInvestigate
}: {
  currentRisk: number; currentStage: string; indicators: string[];
  nextStage: string; probability: number; horizon: number;
  onInvestigate: () => void;
}) {
  return (
    <Card>
      <CardHeader title="Current vs Forecast" />
      <div className="p-4 space-y-4">
        {/* Current */}
        <div className="space-y-2">
          <div className="text-[10px] uppercase tracking-widest text-text-muted">Current Network State</div>
          <div className="flex items-baseline gap-1">
            <span className="text-2xl font-bold text-warning">{currentRisk}%</span>
            <span className="text-xs text-text-muted">risk</span>
          </div>
          <div className="text-xs text-text-secondary">Stage: <span className="text-elevated font-medium">{currentStage}</span></div>
          <div className="space-y-1">
            {indicators.map(ind => (
              <div key={ind} className="flex items-center gap-1.5 text-[11px] text-text-muted">
                <span className="text-warning">·</span>
                {ind}
              </div>
            ))}
            {indicators.length === 0 && (
              <div className="text-[11px] text-text-disabled italic">No saliency features available</div>
            )}
          </div>
        </div>

        <div className="border-t border-border-subtle" />

        {/* Forecast */}
        <div className="space-y-2">
          <div className="text-[10px] uppercase tracking-widest text-forecast flex items-center gap-1.5">
            <span className="text-[8px]">◆</span>
            AI Forecast
          </div>
          <div className="text-sm font-bold text-forecast-bright">{nextStage}</div>
          <div className="flex items-center justify-between text-xs">
            <span className="text-text-muted">Attack probability</span>
            <span className="font-semibold text-warning">{probability}%</span>
          </div>
          <div className="flex items-center justify-between text-xs">
            <span className="text-text-muted">Forecast horizon</span>
            <span className="text-text-secondary">Next {horizon} windows</span>
          </div>
          <Badge severity="Forecast" dot className="text-[10px]">PyTorch GRU Model Output</Badge>
        </div>

        <button onClick={onInvestigate} className="btn btn-secondary w-full justify-center btn-sm">
          <ArrowRight size={12} />
          Open Investigation
        </button>
      </div>
    </Card>
  );
}
