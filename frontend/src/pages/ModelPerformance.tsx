import React, { useState, useEffect } from 'react';
import { AppShell } from '../components/layout/AppShell';
import { Card, CardHeader } from '../components/common/Card';
import { Badge } from '../components/common/Badge';
import { getModelPerformance } from '../api/modelApi';
import { Brain, BarChart2, Activity } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { formatPct } from '../utils/formatters';
import { cn } from '../utils/cn';

export default function ModelPerformance() {
  const [models, setModels] = useState<any[]>([]);

  useEffect(() => {
    getModelPerformance().then(setModels);
  }, []);

  const chartData = models.map(m => ({
    name: m.id.toUpperCase(),
    BinaryAUC: m.binaryAuc * 100,
    BinaryF1: m.binaryF1 * 100,
  }));

  return (
    <AppShell>
      <div className="mb-6">
        <h2 className="text-xl font-bold text-text-primary tracking-tight">Model Performance Benchmark</h2>
        <p className="text-sm text-text-muted mt-1">Comparison of baseline models vs. the temporal GRU/LSTM World Models.</p>
        <Badge severity="Info" dot className="mt-2 text-[10px]">EVALUATION: CIC-IDS2017 TEST SET</Badge>
      </div>

      {/* Primary Model Highlight */}
      {models.filter(m => m.primary).map(primary => (
        <Card key={primary.id} glow="forecast" className="mb-4">
          <CardHeader title="Primary Forecasting Model" icon={<Brain size={14} />} />
          <div className="p-6 grid grid-cols-1 lg:grid-cols-4 gap-6 items-center bg-forecast-bg/10">
            <div className="lg:col-span-2 space-y-2">
              <h3 className="text-xl font-bold text-forecast-bright">{primary.name}</h3>
              <p className="text-sm text-text-secondary">{primary.description}</p>
              <div className="pt-2 flex flex-wrap gap-2">
                <Badge severity="Forecast" className="py-0.5">Parameters: {(primary.parameters/1000).toFixed(1)}k</Badge>
                <Badge severity="Forecast" className="py-0.5">Latency: {primary.inferenceMs}ms</Badge>
              </div>
            </div>
            <div className="grid grid-cols-2 gap-4 lg:col-span-2">
              <div className="border border-forecast-dim bg-surface-2 rounded p-3">
                <div className="text-[10px] uppercase text-text-muted mb-1">Binary AUC (k=1)</div>
                <div className="text-2xl font-bold text-text-primary">{primary.binaryAuc.toFixed(4)}</div>
              </div>
              <div className="border border-forecast-dim bg-surface-2 rounded p-3">
                <div className="text-[10px] uppercase text-text-muted mb-1">Binary F1</div>
                <div className="text-2xl font-bold text-text-primary">{primary.binaryF1.toFixed(4)}</div>
              </div>
            </div>
          </div>
        </Card>
      ))}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-4">
        <Card>
          <CardHeader title="Model Benchmark Table" icon={<Activity size={14} />} />
          <div className="overflow-x-auto">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Binary AUC</th>
                  <th>Binary F1</th>
                  <th>Precision</th>
                  <th>Recall</th>
                  <th>Latency</th>
                </tr>
              </thead>
              <tbody>
                {models.map(m => (
                  <tr key={m.id} className={cn(m.primary && 'bg-forecast-dim/10')}>
                    <td className="font-medium text-text-primary">
                      {m.name}
                      {m.primary && <span className="ml-2 text-[9px] text-forecast border border-forecast px-1 rounded uppercase">Active</span>}
                    </td>
                    <td className="font-mono text-xs text-info">{m.binaryAuc.toFixed(4)}</td>
                    <td className="font-mono text-xs">{m.binaryF1.toFixed(4)}</td>
                    <td className="font-mono text-xs">{m.precision.toFixed(2)}</td>
                    <td className="font-mono text-xs">{m.recall.toFixed(2)}</td>
                    <td className="font-mono text-xs text-text-muted">{m.inferenceMs}ms</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>

        <Card>
          <CardHeader title="Performance Comparison" icon={<BarChart2 size={14} />} />
          <div className="p-4" style={{ height: 260 }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} margin={{ left: -15, right: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e2a3a" vertical={false} />
                <XAxis dataKey="name" tick={{ fill: '#64748b', fontSize: 10 }} axisLine={false} tickLine={false} />
                <YAxis domain={[50, 100]} tick={{ fill: '#64748b', fontSize: 10 }} axisLine={false} tickLine={false} tickFormatter={v => `${v}%`} />
                <Tooltip contentStyle={{ background: '#171d27', border: '1px solid #243040', fontSize: 12 }} />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Bar dataKey="BinaryAUC" fill="#38bdf8" name="Binary AUC" radius={[2, 2, 0, 0]} maxBarSize={40} />
                <Bar dataKey="BinaryF1" fill="#a78bfa" name="Binary F1" radius={[2, 2, 0, 0]} maxBarSize={40} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>
      </div>
    </AppShell>
  );
}
