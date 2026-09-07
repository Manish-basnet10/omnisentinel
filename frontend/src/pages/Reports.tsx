import React, { useState } from 'react';
import { AppShell } from '../components/layout/AppShell';
import { Card, CardHeader } from '../components/common/Card';
import { generateReport } from '../api/modelApi';
import { FileText, Download, Loader, Calendar, Filter } from 'lucide-react';
import { cn } from '../utils/cn';

export default function Reports() {
  const [generating, setGenerating] = useState(false);
  const [reportResult, setReportResult] = useState<any>(null);
  const [reportType, setReportType] = useState('Security Summary');

  async function handleGenerate(e: React.FormEvent) {
    e.preventDefault();
    setGenerating(true);
    setReportResult(null);
    const res = await generateReport(reportType, { from: '2026-09-01', to: '2026-09-07' });
    setReportResult(res);
    setGenerating(false);
  }

  return (
    <AppShell>
      <div className="mb-6">
        <h2 className="text-xl font-bold text-text-primary tracking-tight">Reports & Exports</h2>
        <p className="text-sm text-text-muted mt-1">Generate comprehensive network security and AI forecast summaries.</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader title="Generate Report" icon={<FileText size={14} />} />
          <form onSubmit={handleGenerate} className="p-5 space-y-5">
            <div className="space-y-4">
              <div>
                <label className="block text-xs text-text-secondary mb-1.5">Report Type</label>
                <select value={reportType} onChange={e => setReportType(e.target.value)} className="select w-full">
                  <option>Security Summary</option>
                  <option>AI Forecast Analysis</option>
                  <option>Investigation Export (PP-1042)</option>
                  <option>Compliance & Audit</option>
                </select>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-xs text-text-secondary mb-1.5">Date Range From</label>
                  <div className="relative">
                    <Calendar size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-text-disabled" />
                    <input type="date" defaultValue="2026-09-01" className="input pl-9" />
                  </div>
                </div>
                <div>
                  <label className="block text-xs text-text-secondary mb-1.5">Date Range To</label>
                  <div className="relative">
                    <Calendar size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-text-disabled" />
                    <input type="date" defaultValue="2026-09-07" className="input pl-9" />
                  </div>
                </div>
              </div>

              <div>
                <label className="block text-xs text-text-secondary mb-1.5 flex items-center gap-2">
                  <Filter size={12} /> Filters
                </label>
                <div className="space-y-2 p-3 bg-surface-2 rounded border border-border-subtle">
                  <label className="flex items-center gap-2 text-sm text-text-primary">
                    <input type="checkbox" defaultChecked className="w-3.5 h-3.5 accent-forecast bg-surface-4" /> Include AI Forecasts
                  </label>
                  <label className="flex items-center gap-2 text-sm text-text-primary">
                    <input type="checkbox" defaultChecked className="w-3.5 h-3.5 accent-forecast bg-surface-4" /> Include High/Critical Alerts only
                  </label>
                  <label className="flex items-center gap-2 text-sm text-text-primary">
                    <input type="checkbox" defaultChecked className="w-3.5 h-3.5 accent-forecast bg-surface-4" /> Include MITRE ATT&CK mapping
                  </label>
                </div>
              </div>
            </div>

            <button type="submit" disabled={generating} className="btn btn-primary w-full justify-center">
              {generating ? <Loader size={14} className="animate-spin" /> : <FileText size={14} />}
              {generating ? 'Compiling data…' : 'Generate Report'}
            </button>
          </form>
        </Card>

        {reportResult ? (
          <Card glow="forecast">
            <CardHeader title="Report Ready" icon={<Download size={14} />} right={<span className="badge badge-forecast">Generated</span>} />
            <div className="p-6 space-y-6">
              <div>
                <div className="text-[10px] uppercase text-text-muted mb-1">Report ID</div>
                <div className="font-mono text-sm text-text-primary">{reportResult.id}</div>
              </div>
              
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <div className="text-[10px] uppercase text-text-muted mb-1">Type</div>
                  <div className="text-sm font-semibold">{reportResult.type}</div>
                </div>
                <div>
                  <div className="text-[10px] uppercase text-text-muted mb-1">Timestamp</div>
                  <div className="text-xs text-text-secondary font-mono">{reportResult.generatedAt}</div>
                </div>
              </div>

              <div className="flex flex-col gap-3 pt-4 border-t border-border-subtle">
                <button className="btn btn-secondary justify-center text-forecast-bright border-forecast-dim bg-forecast-bg/30 hover:bg-forecast hover:text-surface-0">
                  <Download size={14} /> Download PDF
                </button>
                <button className="btn btn-secondary justify-center">
                  <Download size={14} /> Export Raw CSV (Demo)
                </button>
              </div>
            </div>
          </Card>
        ) : (
          <div className="h-full border-2 border-dashed border-border-emphasis rounded-lg flex flex-col items-center justify-center text-center p-8 bg-surface-2 opacity-60">
            <FileText size={32} className="text-text-disabled mb-3" />
            <div className="text-sm text-text-secondary font-medium">No report generated</div>
            <div className="text-xs text-text-muted mt-1">Configure options and click Generate Report.</div>
          </div>
        )}
      </div>
    </AppShell>
  );
}
