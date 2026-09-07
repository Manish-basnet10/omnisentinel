import React, { useEffect, useState, useCallback } from 'react';
import { AppShell } from '../components/layout/AppShell';
import { KpiCard } from '../components/dashboard/KpiCard';
import { ForecastHeroCard } from '../components/dashboard/ForecastHero';
import { AttackTrajectory } from '../components/dashboard/AttackTrajectory';
import { RiskForecastChart } from '../components/dashboard/RiskForecastChart';
import { ExplainabilityCard } from '../components/dashboard/ExplainabilityCard';
import { TrafficOverview } from '../components/dashboard/TrafficOverview';
import { SkeletonCard } from '../components/common/States';
import { Card, CardHeader } from '../components/common/Card';
import { Badge } from '../components/common/Badge';
import { getDashboardSummary, getForecastHero, getAttackTrajectory, getRiskTimeSeries, getExplainability, getTrafficOverview } from '../api/dashboardApi';
import { mockTrafficOverview } from '../data/mockDashboard';
import {
  AlertTriangle, Activity, BarChart2, Zap, Shield,
  Radio, ArrowRight,
} from 'lucide-react';
import { formatNumber, riskLevelFromScore } from '../utils/formatters';
import { useNavigate } from 'react-router-dom';

export default function Dashboard() {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(true);
  const [summary, setSummary]       = useState<any>(null);
  const [forecast, setForecast]     = useState<any>(null);
  const [trajectory, setTrajectory] = useState<any[]>([]);
  const [riskSeries, setRiskSeries] = useState<any[]>([]);
  const [explain, setExplain]       = useState<any>(null);
  const [traffic, setTraffic]       = useState<any>(mockTrafficOverview);

  const loadAll = useCallback(async () => {
    const [s, f, t, r, e] = await Promise.all([
      getDashboardSummary(),
      getForecastHero(),
      getAttackTrajectory(),
      getRiskTimeSeries(),
      getExplainability(),
    ]);
    setSummary(s);
    setForecast(f);
    setTrajectory(t);
    setRiskSeries(r);
    setExplain(e);
    setLoading(false);
  }, []);

  useEffect(() => {
    loadAll();
    // Live metric refresh every 8s
    const id = setInterval(async () => {
      const s = await getDashboardSummary();
      setSummary(s);
    }, 8000);
    return () => clearInterval(id);
  }, [loadAll]);

  if (loading) {
    return (
      <AppShell>
        <div className="grid grid-cols-5 gap-3 mb-4">
          {Array.from({ length: 5 }).map((_, i) => <SkeletonCard key={i} rows={2} />)}
        </div>
        <SkeletonCard rows={6} />
      </AppShell>
    );
  }

  const riskLevel = riskLevelFromScore(summary.networkRisk);

  return (
    <AppShell alertCount={summary.activeThreats}>
      {/* KPI Row */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3 mb-4">
        <KpiCard
          title="Network Risk"
          value={summary.networkRisk}
          unit="%"
          delta={summary.networkRiskDelta}
          deltaLabel={`${summary.networkRiskDelta > 0 ? '+' : ''}${summary.networkRiskDelta}pp from previous window`}
          severity={summary.networkRisk >= 80 ? 'critical' : summary.networkRisk >= 65 ? 'warning' : 'elevated'}
          icon={<AlertTriangle size={16} />}
          subtitle={riskLevel}
        />
        <KpiCard
          title="Active Threats"
          value={summary.activeThreats}
          severity="warning"
          icon={<Activity size={16} />}
          subtitle="Require investigation"
        />
        <KpiCard
          title="Suspicious Flows"
          value={formatNumber(summary.suspiciousFlows)}
          severity="elevated"
          icon={<BarChart2 size={16} />}
          subtitle="Last observation window"
        />
        <KpiCard
          title="Forecast Confidence"
          value={summary.forecastConfidence}
          unit="%"
          severity="forecast"
          icon={<Zap size={16} />}
          subtitle="Model estimate"
        />
        <KpiCard
          title="Protected Assets"
          value={summary.protectedAssets}
          severity="safe"
          icon={<Shield size={16} />}
          subtitle="Active coverage"
        />
      </div>

      {/* Hero Forecast */}
      <div className="mb-4">
        <ForecastHeroCard forecast={forecast} onRefresh={setForecast} />
      </div>

      {/* Attack Trajectory */}
      <div className="mb-4">
        <AttackTrajectory stages={trajectory} />
      </div>

      {/* Risk Chart + Current vs Forecast */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 mb-4">
        <div className="lg:col-span-2">
          <RiskForecastChart data={riskSeries} />
        </div>
        <CurrentVsForecastCard
          currentRisk={72}
          currentStage="Reconnaissance"
          indicators={['Port scanning', 'High SYN ratio', 'Unusual destination count']}
          nextStage="Initial Access"
          probability={82}
          horizon={5}
          onInvestigate={() => navigate('/investigations')}
        />
      </div>

      {/* Explainability + Traffic */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <ExplainabilityCard data={explain} />
        <TrafficOverview data={traffic} />
      </div>
    </AppShell>
  );
}

// ── Current vs Forecast side-by-side card ───────────────
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
            <span className="text-text-muted">Estimated probability</span>
            <span className="font-semibold text-warning">{probability}%</span>
          </div>
          <div className="flex items-center justify-between text-xs">
            <span className="text-text-muted">Forecast horizon</span>
            <span className="text-text-secondary">Next {horizon} windows</span>
          </div>
          <Badge severity="Forecast" dot className="text-[10px]">Model Estimate — Demo Data</Badge>
        </div>

        <button onClick={onInvestigate} className="btn btn-secondary w-full justify-center btn-sm">
          <ArrowRight size={12} />
          Open Investigation
        </button>
      </div>
    </Card>
  );
}
