import React, { useState, useCallback } from 'react';
import { AppShell } from '../components/layout/AppShell';
import { Card, CardHeader } from '../components/common/Card';
import { ProgressBar } from '../components/common/States';
import { Badge } from '../components/common/Badge';
import { uploadTrafficFile, getProtocolDistribution, getTcpFlags, getTopPorts } from '../api/networkApi';
import { mockProtocolDistribution, mockTcpFlags, mockTopPorts } from '../data/mockTraffic';
import { mockTrafficOverview } from '../data/mockDashboard';
import { TrafficOverview } from '../components/dashboard/TrafficOverview';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
  PieChart, Pie, Cell, Legend,
} from 'recharts';
import { Upload, CheckCircle, Loader, BarChart2, FileUp } from 'lucide-react';
import { cn } from '../utils/cn';

const PIE_COLORS = ['#38bdf8','#a78bfa','#f97316','#22c55e','#eab308','#ef4444','#64748b'];

const WORKFLOW_STEPS = ['Upload','Extract','Normalize','Analyze','Forecast','Explain'];

export default function TrafficAnalysis() {
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadStage, setUploadStage]       = useState('');
  const [uploading, setUploading]           = useState(false);
  const [uploadDone, setUploadDone]         = useState(false);
  const [uploadMsg, setUploadMsg]           = useState('');
  const [dragOver, setDragOver]             = useState(false);

  const handleFile = useCallback(async (file: File) => {
    const allowed = ['.csv', '.pcap', '.pcapng'];
    const ext = file.name.substring(file.name.lastIndexOf('.')).toLowerCase();
    if (!allowed.includes(ext)) {
      setUploadMsg(`Unsupported format. Allowed: ${allowed.join(', ')}`);
      return;
    }
    setUploading(true);
    setUploadDone(false);
    setUploadProgress(0);
    const result = await uploadTrafficFile(file, (pct, stage) => {
      setUploadProgress(pct);
      setUploadStage(stage);
    });
    setUploading(false);
    setUploadDone(true);
    setUploadMsg(result.message);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  }, [handleFile]);

  const currentWorkflowIdx = WORKFLOW_STEPS.indexOf(uploadStage);

  return (
    <AppShell>
      {/* Upload section */}
      <div className="mb-4">
        <Card>
          <CardHeader title="Traffic File Upload" subtitle="Upload .csv, .pcap, or .pcapng for AI analysis" icon={<FileUp size={14} />} />
          <div className="p-4 space-y-4">
            {/* Workflow steps */}
            <div className="flex items-center gap-0 overflow-x-auto">
              {WORKFLOW_STEPS.map((step, i) => {
                const done    = uploadDone || (uploading && currentWorkflowIdx > i);
                const current = uploading && currentWorkflowIdx === i;
                return (
                  <React.Fragment key={step}>
                    <div className={cn(
                      'flex items-center gap-1.5 px-3 py-1.5 rounded text-[11px] font-medium whitespace-nowrap',
                      done    ? 'bg-safe-bg text-safe border border-safe-dim' :
                      current ? 'bg-forecast-bg text-forecast border border-forecast' :
                                'bg-surface-3 text-text-disabled border border-border-subtle'
                    )}>
                      {done && <CheckCircle size={11} />}
                      {current && <Loader size={11} className="animate-spin" />}
                      {step}
                    </div>
                    {i < WORKFLOW_STEPS.length - 1 && (
                      <div className={cn('h-px w-6 flex-shrink-0', done ? 'bg-safe' : 'bg-border-subtle')} />
                    )}
                  </React.Fragment>
                );
              })}
            </div>

            {/* Drop zone */}
            <div
              onDragOver={e => { e.preventDefault(); setDragOver(true); }}
              onDragLeave={() => setDragOver(false)}
              onDrop={handleDrop}
              className={cn(
                'border-2 border-dashed rounded-lg p-8 text-center transition-colors cursor-pointer',
                dragOver ? 'border-forecast bg-forecast-bg' : 'border-border-emphasis hover:border-forecast-dim bg-surface-2'
              )}
              onClick={() => document.getElementById('file-input')?.click()}
              role="button"
              tabIndex={0}
              aria-label="Upload traffic file"
            >
              <input
                id="file-input"
                type="file"
                accept=".csv,.pcap,.pcapng"
                hidden
                onChange={e => { const f = e.target.files?.[0]; if (f) handleFile(f); }}
              />
              <Upload size={28} className="mx-auto mb-3 text-text-muted" />
              <p className="text-sm text-text-secondary">
                Drag & drop or <span className="text-forecast">browse</span> to upload
              </p>
              <p className="text-[11px] text-text-muted mt-1">Supports .csv, .pcap, .pcapng · PCAP parsing done server-side</p>
            </div>

            {/* Progress */}
            {uploading && (
              <div className="space-y-2">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-text-secondary">{uploadStage}…</span>
                  <span className="text-text-muted font-mono">{uploadProgress}%</span>
                </div>
                <ProgressBar value={uploadProgress} color="forecast" />
              </div>
            )}

            {uploadDone && (
              <div className="flex items-center gap-2 text-safe text-sm">
                <CheckCircle size={15} />
                <span>{uploadMsg}</span>
              </div>
            )}
            {uploadMsg && !uploadDone && (
              <div className="text-critical text-sm">{uploadMsg}</div>
            )}
          </div>
        </Card>
      </div>

      {/* Charts row */}
      <TrafficOverview data={mockTrafficOverview} />

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 mt-4">
        {/* Protocol Distribution */}
        <Card>
          <CardHeader title="Protocol Distribution" />
          <div className="p-4" style={{ height: 220 }}>
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie data={mockProtocolDistribution} cx="50%" cy="50%" outerRadius={75} dataKey="value" nameKey="name">
                  {mockProtocolDistribution.map((_, i) => (
                    <Cell key={i} fill={PIE_COLORS[i % PIE_COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip contentStyle={{ background: '#171d27', border: '1px solid #243040', fontSize: 11 }} />
                <Legend wrapperStyle={{ fontSize: 11 }} />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </Card>

        {/* Top Destination Ports */}
        <Card>
          <CardHeader title="Top Destination Ports" />
          <div className="p-4" style={{ height: 220 }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={mockTopPorts.destination} layout="vertical" margin={{ left: 10, right: 10 }}>
                <CartesianGrid strokeDasharray="2 4" stroke="#1e2a3a" horizontal={false} />
                <XAxis type="number" tick={{ fill: '#64748b', fontSize: 10 }} axisLine={false} tickLine={false} />
                <YAxis type="category" dataKey="port" tick={{ fill: '#64748b', fontSize: 10 }} axisLine={false} tickLine={false} />
                <Tooltip contentStyle={{ background: '#171d27', border: '1px solid #243040', fontSize: 11 }} />
                <Bar dataKey="count" fill="#38bdf8" radius={[0, 3, 3, 0]} maxBarSize={16} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>

        {/* TCP Flag Distribution */}
        <Card>
          <CardHeader title="TCP Flag Distribution" />
          <div className="p-4" style={{ height: 220 }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={mockTcpFlags} margin={{ left: -10, right: 8 }}>
                <CartesianGrid strokeDasharray="2 4" stroke="#1e2a3a" vertical={false} />
                <XAxis dataKey="flag" tick={{ fill: '#64748b', fontSize: 10 }} axisLine={false} tickLine={false} />
                <YAxis tick={{ fill: '#64748b', fontSize: 10 }} axisLine={false} tickLine={false} />
                <Tooltip contentStyle={{ background: '#171d27', border: '1px solid #243040', fontSize: 11 }} />
                <Bar dataKey="count" fill="#a78bfa" radius={[3, 3, 0, 0]} maxBarSize={24} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>
      </div>
    </AppShell>
  );
}
