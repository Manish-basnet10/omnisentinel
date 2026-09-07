import React, { useState, useEffect } from 'react';
import { AppShell } from '../components/layout/AppShell';
import { Card, CardHeader } from '../components/common/Card';
import { ProgressBar, SkeletonCard } from '../components/common/States';
import { getModelExplainability } from '../api/modelApi';
import { LineChart, FlaskConical, Target, AlertTriangle } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';

export default function Explainability() {
  const [data, setData] = useState<any>(null);

  useEffect(() => {
    getModelExplainability().then(setData);
  }, []);

  if (!data) return <AppShell><SkeletonCard rows={8} /></AppShell>;

  const globalData = data.globalImportance.slice(0, 10).map((d: any) => ({
    name: d.feature,
    value: d.importance * 100
  }));

  const predExp = data.predictionExplanation;

  return (
    <AppShell>
      <div className="mb-6">
        <h2 className="text-xl font-bold text-text-primary tracking-tight">AI Explainability</h2>
        <p className="text-sm text-text-muted mt-1">Understanding model feature saliency and prediction reasoning.</p>
        <div className="mt-2 flex items-center gap-2 text-[10px] text-text-muted">
          <FlaskConical size={12} />
          <span>Demo Data — Simulated feature contribution estimates. Not real model output.</span>
        </div>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-4 mb-4">
        {/* Local Explainability (Current Prediction) */}
        <Card glow="forecast">
          <CardHeader title="Current Prediction Analysis" icon={<Target size={14} />} />
          <div className="p-6">
            <div className="flex items-end justify-between border-b border-border-subtle pb-4 mb-4">
              <div>
                <div className="text-[10px] uppercase text-text-muted tracking-widest mb-1">Target State</div>
                <div className="text-xl font-bold text-forecast-bright">{predExp.prediction}</div>
              </div>
              <div className="text-right">
                <div className="text-[10px] uppercase text-text-muted tracking-widest mb-1">Confidence</div>
                <div className="text-2xl font-mono font-bold text-text-primary">{predExp.confidence}%</div>
              </div>
            </div>

            <div className="space-y-6">
              <div>
                <h4 className="text-xs font-semibold text-text-primary mb-3 flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full bg-warning" /> Positive Influencers (Drives probability UP)
                </h4>
                <div className="space-y-3 pl-4">
                  {predExp.positiveFactors.map((f: any) => (
                    <div key={f.feature} className="space-y-1">
                      <div className="flex justify-between text-xs">
                        <span className="text-text-secondary">{f.feature}</span>
                        <span className="font-mono text-warning">+{Math.round(f.contribution * 100)}%</span>
                      </div>
                      <ProgressBar value={f.contribution * 100} color="warning" />
                    </div>
                  ))}
                </div>
              </div>

              <div>
                <h4 className="text-xs font-semibold text-text-primary mb-3 flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full bg-info" /> Negative Influencers (Drives probability DOWN)
                </h4>
                <div className="space-y-3 pl-4">
                  {predExp.negativeFactors.map((f: any) => (
                    <div key={f.feature} className="space-y-1">
                      <div className="flex justify-between text-xs">
                        <span className="text-text-secondary">{f.feature}</span>
                        <span className="font-mono text-info">{Math.round(f.contribution * 100)}%</span>
                      </div>
                      <ProgressBar value={Math.abs(f.contribution * 100)} color="info" />
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </Card>

        {/* Global Feature Importance */}
        <Card>
          <CardHeader title="Global Feature Importance" subtitle="Top 10 features driving model decisions across all states" icon={<LineChart size={14} />} />
          <div className="p-4" style={{ height: 400 }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={globalData} layout="vertical" margin={{ left: 40, right: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e2a3a" horizontal={false} />
                <XAxis type="number" domain={[0, 'dataMax + 2']} tick={{ fill: '#64748b', fontSize: 10 }} axisLine={false} tickLine={false} tickFormatter={v => `${v}%`} />
                <YAxis type="category" dataKey="name" tick={{ fill: '#64748b', fontSize: 10 }} axisLine={false} tickLine={false} width={120} />
                <Tooltip contentStyle={{ background: '#171d27', border: '1px solid #243040', fontSize: 11 }} />
                <Bar dataKey="value" name="Importance" radius={[0, 4, 4, 0]} barSize={20}>
                  {globalData.map((_: any, index: number) => (
                    <Cell key={`cell-${index}`} fill={index < 3 ? '#a78bfa' : '#38bdf8'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>
      </div>

      <Card>
        <CardHeader title="Confusion Matrix (Demo Data)" subtitle="Multiclass head performance on unseen attack types" icon={<AlertTriangle size={14} />} />
        <div className="p-6 overflow-x-auto">
          <table className="w-full text-center border-collapse text-xs">
            <thead>
              <tr>
                <th className="p-2 border border-border-subtle bg-surface-2 text-text-muted font-medium w-24">True \ Pred</th>
                {data.confusionMatrix.labels.map((l: string) => <th key={l} className="p-2 border border-border-subtle bg-surface-2 text-text-secondary font-medium w-20">{l}</th>)}
              </tr>
            </thead>
            <tbody>
              {data.confusionMatrix.matrix.map((row: number[], i: number) => (
                <tr key={i}>
                  <td className="p-2 border border-border-subtle bg-surface-2 text-text-secondary font-medium text-left">{data.confusionMatrix.labels[i]}</td>
                  {row.map((val: number, j: number) => {
                    const isDiagonal = i === j;
                    const intensity = Math.min(1, val / 2000); // Hacky normalization for visual
                    const bgClass = isDiagonal 
                      ? (val > 500 ? 'bg-forecast/30 text-forecast-bright' : 'bg-forecast/10 text-text-primary')
                      : (val > 10 ? 'bg-critical-bg/40 text-warning' : 'bg-transparent text-text-muted');

                    return (
                      <td key={j} className={`p-2 border border-border-subtle ${bgClass} font-mono`}>
                        {val}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

    </AppShell>
  );
}
