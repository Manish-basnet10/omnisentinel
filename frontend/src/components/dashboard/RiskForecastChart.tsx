import React from 'react';
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip,
  ReferenceLine, ResponsiveContainer, Legend,
} from 'recharts';
import { Card, CardHeader } from '../common/Card';
import { TrendingUp } from 'lucide-react';
import type { mockRiskTimeSeries } from '../../data/mockDashboard';

type DataPoint = typeof mockRiskTimeSeries[number];

const CustomTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null;
  const d: DataPoint = payload[0]?.payload;
  const isForecast = d?.type === 'forecast';
  return (
    <div className="bg-surface-4 border border-border-emphasis rounded p-3 text-xs space-y-1 shadow-xl">
      <div className="text-text-muted font-mono">{label}</div>
      <div className="flex items-center gap-2">
        <span className="text-text-secondary">Risk Score:</span>
        <span className={`font-semibold ${d.risk >= 80 ? 'text-critical' : d.risk >= 65 ? 'text-warning' : d.risk >= 45 ? 'text-elevated' : 'text-safe'}`}>
          {d.risk}%
        </span>
      </div>
      <div className="flex items-center gap-2">
        <span className="text-text-secondary">Stage:</span>
        <span className="text-text-primary">{d.stage}</span>
      </div>
      {isForecast && (
        <div className="text-forecast text-[10px] pt-1 border-t border-border-subtle">
          ◆ Model Estimate — Demo Data
        </div>
      )}
    </div>
  );
};

export function RiskForecastChart({ data }: { data: DataPoint[] }) {
  const nowIdx = data.findIndex(d => d.type === 'now');
  const nowLabel = nowIdx >= 0 ? data[nowIdx].t : undefined;

  // Split data for historical vs forecast styling
  const historical = data.filter(d => d.type !== 'forecast');
  const forecast   = data.filter(d => d.type !== 'historical');

  return (
    <Card>
      <CardHeader
        title="Attack Risk Forecast"
        subtitle="Historical risk progression and AI-predicted trajectory"
        icon={<TrendingUp size={14} />}
        right={
          <div className="flex items-center gap-3 text-[10px] text-text-muted">
            <div className="flex items-center gap-1.5">
              <div className="w-6 h-0.5 bg-info" />
              <span>Observed</span>
            </div>
            <div className="flex items-center gap-1.5">
              <div className="w-6 h-0.5 bg-forecast border-dashed" style={{ borderBottom: '2px dashed #a78bfa', background: 'none' }} />
              <span>Forecast</span>
            </div>
          </div>
        }
      />
      <div className="p-4" style={{ height: 260 }}>
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data} margin={{ top: 8, right: 8, left: -10, bottom: 0 }}>
            <defs>
              <linearGradient id="riskGradHist" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%"  stopColor="#38bdf8" stopOpacity={0.25} />
                <stop offset="95%" stopColor="#38bdf8" stopOpacity={0.02} />
              </linearGradient>
              <linearGradient id="riskGradFcast" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%"  stopColor="#a78bfa" stopOpacity={0.3} />
                <stop offset="95%" stopColor="#a78bfa" stopOpacity={0.02} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="2 4" stroke="#1e2a3a" vertical={false} />
            <XAxis dataKey="t" tick={{ fill: '#64748b', fontSize: 10 }} axisLine={false} tickLine={false} />
            <YAxis domain={[0, 100]} tickFormatter={v => `${v}%`} tick={{ fill: '#64748b', fontSize: 10 }} axisLine={false} tickLine={false} />
            <Tooltip content={<CustomTooltip />} />

            {/* Historical area */}
            <Area
              data={historical}
              type="monotone"
              dataKey="risk"
              stroke="#38bdf8"
              strokeWidth={2}
              fill="url(#riskGradHist)"
              dot={false}
              activeDot={{ r: 4, fill: '#38bdf8' }}
            />
            {/* Forecast area */}
            <Area
              data={forecast}
              type="monotone"
              dataKey="risk"
              stroke="#a78bfa"
              strokeWidth={2}
              strokeDasharray="5 3"
              fill="url(#riskGradFcast)"
              dot={false}
              activeDot={{ r: 4, fill: '#a78bfa' }}
            />

            {/* NOW marker */}
            {nowLabel && (
              <ReferenceLine
                x={nowLabel}
                stroke="#eab308"
                strokeWidth={1.5}
                strokeDasharray="4 2"
                label={{ value: 'NOW', position: 'insideTopLeft', fill: '#eab308', fontSize: 10 }}
              />
            )}
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </Card>
  );
}
