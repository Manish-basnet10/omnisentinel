import React, { useRef, useState } from 'react';
import { AppShell } from '../components/layout/AppShell';
import { Card, CardHeader } from '../components/common/Card';
import { Badge } from '../components/common/Badge';
import { TrafficOverview } from '../components/dashboard/TrafficOverview';
import { useAnalysis } from '../hooks/useAnalysis';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
  PieChart, Pie, Cell, Legend,
} from 'recharts';
import { Upload, CheckCircle, BarChart2, FileUp } from 'lucide-react';
import { cn } from '../utils/cn';

const PIE_COLORS = ['#38bdf8','#a78bfa','#f97316','#22c55e','#eab308','#ef4444','#64748b'];

export default function TrafficAnalysis() {
  const { analysis, uploadDataset, isProcessing } = useAnalysis();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    if (e.dataTransfer.files?.[0]) uploadDataset(e.dataTransfer.files[0]);
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.[0]) uploadDataset(e.target.files[0]);
    e.target.value = '';
  };

  // Mock charts for protocol/TCP flags (since backend doesn't return these for CSV/Parquet yet)
  const protocols = [
    { name: 'TCP', value: 75.2 },
    { name: 'UDP', value: 20.1 },
    { name: 'ICMP', value: 3.5 },
    { name: 'Other', value: 1.2 },
  ];
  const tcpFlags = [
    { flag: 'SYN', count: 12500 },
    { flag: 'ACK', count: 45000 },
    { flag: 'FIN', count: 8200 },
    { flag: 'RST', count: 1500 },
    { flag: 'PSH', count: 21000 },
  ];

  return (
    <AppShell>
      <div className="mb-6">
        <h2 className="text-xl font-bold text-text-primary tracking-tight">Traffic Analysis</h2>
        <p className="text-sm text-text-muted mt-1">Upload network traffic files and review flow-level characteristics.</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
        <div className="lg:col-span-1">
          <Card className="h-full">
            <CardHeader title="Dataset Upload" icon={<FileUp size={14} />} />
            <div className="p-6 flex flex-col items-center justify-center h-[280px]">
              <div
                className={cn(
                  'w-full h-full border-2 border-dashed rounded-xl flex flex-col items-center justify-center p-6 text-center transition-all',
                  dragOver ? 'border-forecast bg-forecast-bg/30' : 'border-border-subtle hover:border-text-muted',
                  isProcessing && 'opacity-50 pointer-events-none'
                )}
                onDragOver={e => { e.preventDefault(); setDragOver(true); }}
                onDragLeave={() => setDragOver(false)}
                onDrop={handleDrop}
              >
                <Upload size={32} className={cn('mb-4', dragOver ? 'text-forecast animate-bounce' : 'text-text-disabled')} />
                <h3 className="text-sm font-semibold text-text-primary mb-1">
                  Drag & Drop file here
                </h3>
                <p className="text-xs text-text-muted mb-4 max-w-[200px]">
                  Supports .csv, .parquet, .pcap, .pcapng up to 200MB
                </p>
                <button
                  onClick={() => fileInputRef.current?.click()}
                  className="btn btn-primary"
                  disabled={isProcessing}
                >
                  Browse Files
                </button>
                <input ref={fileInputRef} type="file" accept=".csv,.parquet,.pcap,.pcapng" hidden onChange={handleFileChange} />
              </div>
            </div>
          </Card>
        </div>

        <div className="lg:col-span-2">
          {analysis ? (
            <Card className="h-full">
              <CardHeader title="Analysis Result Summary" icon={<CheckCircle size={14} className="text-safe" />} />
              <div className="p-6 grid grid-cols-2 gap-6">
                <div>
                  <div className="text-xs uppercase text-text-muted mb-1">Dataset</div>
                  <div className="font-mono text-sm text-text-primary mb-4 truncate">{analysis.dataset.filename}</div>

                  <div className="text-xs uppercase text-text-muted mb-1">Model Pipeline</div>
                  <div className="flex gap-2 mb-4">
                    <Badge severity="Forecast">{analysis.model.type}</Badge>
                    <Badge severity="Safe">{analysis.model.features_used} Features</Badge>
                  </div>
                </div>
                <div>
                  <div className="text-xs uppercase text-text-muted mb-1">Network Risk</div>
                  <div className={cn(
                    'text-3xl font-bold mb-1',
                    analysis.prediction.risk_level === 'CRITICAL' ? 'text-critical' :
                    analysis.prediction.risk_level === 'HIGH' ? 'text-warning' :
                    analysis.prediction.risk_level === 'MEDIUM' ? 'text-elevated' : 'text-safe'
                  )}>
                    {analysis.prediction.current_risk.toFixed(1)}%
                  </div>
                  <Badge severity={
                    analysis.prediction.risk_level === 'CRITICAL' ? 'Critical' :
                    analysis.prediction.risk_level === 'HIGH' ? 'Warning' :
                    analysis.prediction.risk_level === 'MEDIUM' ? 'Info' : 'Safe'
                  }>{analysis.prediction.risk_level}</Badge>
                </div>
                <div className="col-span-2 grid grid-cols-3 gap-4 p-4 bg-surface-2 rounded-lg border border-border-subtle mt-2">
                  <div>
                    <div className="text-[10px] uppercase text-text-muted">Flows Analyzed</div>
                    <div className="text-lg font-semibold text-text-primary">{analysis.dataset.rows.toLocaleString()}</div>
                  </div>
                  <div>
                    <div className="text-[10px] uppercase text-text-muted">Predicted Attack Stage</div>
                    <div className="text-sm font-semibold text-text-primary mt-1">{analysis.prediction.predicted_next_stage || 'None'}</div>
                  </div>
                  <div>
                    <div className="text-[10px] uppercase text-text-muted">Confidence</div>
                    <div className="text-lg font-semibold text-forecast-bright">{analysis.prediction.confidence ?? 0}%</div>
                  </div>
                </div>
              </div>
            </Card>
          ) : (
            <Card className="h-full flex flex-col items-center justify-center p-6 text-center text-text-muted border-dashed border-2 border-border-subtle bg-transparent">
              <BarChart2 size={32} className="mb-4 opacity-50" />
              <p className="text-sm">Upload a dataset to see analysis results.</p>
            </Card>
          )}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        <Card>
          <CardHeader title="Protocol Distribution (Estimate)" />
          <div className="h-[260px] w-full p-4">
            <ResponsiveContainer>
              <PieChart>
                <Pie data={protocols} cx="50%" cy="50%" innerRadius={60} outerRadius={80} paddingAngle={2} dataKey="value" stroke="none">
                  {protocols.map((entry, index) => <Cell key={entry.name} fill={PIE_COLORS[index % PIE_COLORS.length]} />)}
                </Pie>
                <Tooltip
                  contentStyle={{ backgroundColor: '#1e1e1e', borderColor: '#333', fontSize: '12px' }}
                  itemStyle={{ color: '#e0e0e0' }}
                  formatter={(val: number) => `${val}%`}
                />
                <Legend verticalAlign="middle" align="right" layout="vertical" iconType="circle" wrapperStyle={{ fontSize: '12px', color: '#a0a0a0' }} />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </Card>

        <Card>
          <CardHeader title="TCP Flags Distribution (Estimate)" />
          <div className="h-[260px] w-full p-4 pt-6">
            <ResponsiveContainer>
              <BarChart data={tcpFlags} margin={{ top: 0, right: 0, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#333" />
                <XAxis dataKey="flag" axisLine={false} tickLine={false} tick={{ fontSize: 11, fill: '#888' }} />
                <YAxis axisLine={false} tickLine={false} tick={{ fontSize: 11, fill: '#888' }} />
                <Tooltip
                  cursor={{ fill: 'rgba(255,255,255,0.05)' }}
                  contentStyle={{ backgroundColor: '#1e1e1e', borderColor: '#333', fontSize: '12px', borderRadius: '8px' }}
                />
                <Bar dataKey="count" fill="#8b5cf6" radius={[4, 4, 0, 0]} barSize={30} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>
      </div>

    </AppShell>
  );
}
