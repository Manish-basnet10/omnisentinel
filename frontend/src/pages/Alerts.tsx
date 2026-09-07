import React, { useState } from 'react';
import { AppShell } from '../components/layout/AppShell';
import { Card, CardHeader } from '../components/common/Card';
import { Badge } from '../components/common/Badge';
import { EmptyState } from '../components/common/States';
import { useAnalysis } from '../hooks/useAnalysis';
import { Bell, Search, Filter, ShieldAlert } from 'lucide-react';
import { updateAlertStatus } from '../api/alertsApi'; // keep this for patching to backend
import { cn } from '../utils/cn';

const SEVERITY_BADGE: Record<string, string> = {
  Critical: 'badge-critical',
  High: 'badge-warning',
  Medium: 'badge-elevated',
  Low: 'badge-info',
};

const STATUS_COLOR: Record<string, string> = {
  New: 'text-warning font-bold',
  Investigating: 'text-forecast font-semibold',
  Acknowledged: 'text-text-secondary',
  Dismissed: 'text-text-disabled line-through',
};

export default function Alerts() {
  const { analysis, refreshAnalysis } = useAnalysis();
  const [filterSeverity, setFilterSeverity] = useState('');
  const [search, setSearch] = useState('');

  if (!analysis) {
    return (
      <AppShell>
        <div className="flex flex-col items-center justify-center h-full min-h-[60vh] text-center">
          <Bell size={48} className="text-text-disabled mb-4" />
          <h2 className="text-xl font-bold text-text-primary mb-2">No Dataset Uploaded</h2>
          <p className="text-text-muted">Upload a dataset to view its associated security alerts.</p>
        </div>
      </AppShell>
    );
  }

  // The analysis session provides the alerts for THIS dataset
  const alerts = analysis.alerts || [];

  async function handleStatusChange(id: string, status: any) {
    await updateAlertStatus(id, status);
    // Refresh the context to get the latest alerts state from backend
    await refreshAnalysis();
  }

  const filtered = alerts.filter(a => {
    const matchSev = !filterSeverity || a.severity === filterSeverity;
    const q = search.toLowerCase();
    const matchQ = !q || (a.title && a.title.toLowerCase().includes(q)) || (a.id && a.id.toLowerCase().includes(q));
    return matchSev && matchQ;
  });

  return (
    <AppShell>
      <Card>
        <CardHeader
          title="Security Alerts"
          subtitle={`Threat notifications for active dataset: ${analysis.dataset.filename}`}
          icon={<Bell size={14} />}
          right={
            <div className="flex items-center gap-2">
              <div className="relative">
                <Search size={12} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-text-muted" />
                <input
                  type="text"
                  placeholder="Search alerts…"
                  value={search}
                  onChange={e => setSearch(e.target.value)}
                  className="input pl-7 h-7 text-xs w-48"
                />
              </div>
              <div className="flex items-center gap-1 bg-surface-3 p-0.5 rounded border border-border-subtle">
                <Filter size={12} className="text-text-muted ml-1" />
                <select
                  value={filterSeverity}
                  onChange={e => setFilterSeverity(e.target.value)}
                  className="select h-6 py-0 pl-1 pr-6 text-xs bg-transparent border-none focus:ring-0"
                >
                  <option value="">All Severities</option>
                  <option value="Critical">Critical</option>
                  <option value="High">High</option>
                  <option value="Medium">Medium</option>
                </select>
              </div>
            </div>
          }
        />
        <div className="overflow-x-auto">
          {filtered.length === 0 ? (
            <div className="p-8">
              <EmptyState
                icon={<ShieldAlert size={24} />}
                title={search || filterSeverity ? 'No matches found' : 'No Alerts'}
                message={search || filterSeverity ? 'Try adjusting your filters.' : 'The model did not generate any high/critical alerts for this dataset.'}
              />
            </div>
          ) : (
            <table className="w-full text-left text-xs">
              <thead className="bg-surface-2 border-b border-border-subtle">
                <tr>
                  <th className="p-3 font-semibold text-text-secondary uppercase tracking-wider w-[120px]">Time</th>
                  <th className="p-3 font-semibold text-text-secondary uppercase tracking-wider">Alert Details</th>
                  <th className="p-3 font-semibold text-text-secondary uppercase tracking-wider">Severity</th>
                  <th className="p-3 font-semibold text-text-secondary uppercase tracking-wider">Status</th>
                  <th className="p-3 font-semibold text-text-secondary uppercase tracking-wider text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border-subtle">
                {filtered.map(alert => (
                  <tr key={alert.id} className="hover:bg-surface-2/50 transition-colors group">
                    <td className="p-3 whitespace-nowrap text-text-muted font-mono">{alert.time}</td>
                    <td className="p-3">
                      <div className="font-semibold text-text-primary mb-0.5 group-hover:text-forecast transition-colors">{alert.title}</div>
                      <div className="text-text-secondary text-[11px] truncate max-w-[300px]" title={alert.detail}>
                        {alert.detail}
                      </div>
                      <div className="flex items-center gap-2 mt-1">
                        <span className="text-[10px] text-text-disabled">ID: {alert.id.substring(0, 8).toUpperCase()}</span>
                        <span className="text-[10px] text-text-disabled">Source: {alert.source}</span>
                      </div>
                    </td>
                    <td className="p-3 whitespace-nowrap">
                      <span className={cn('px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider border', SEVERITY_BADGE[alert.severity])}>
                        {alert.severity}
                      </span>
                    </td>
                    <td className="p-3 whitespace-nowrap">
                      <span className={cn('text-xs flex items-center gap-1.5', STATUS_COLOR[alert.status])}>
                        {alert.status === 'New' && <span className="w-1.5 h-1.5 rounded-full bg-warning animate-pulse" />}
                        {alert.status}
                      </span>
                    </td>
                    <td className="p-3 whitespace-nowrap text-right space-x-2">
                      <select
                        value={alert.status}
                        onChange={e => handleStatusChange(alert.id, e.target.value)}
                        className="select py-1 px-2 text-[10px]"
                      >
                        <option value="New">New</option>
                        <option value="Investigating">Investigate</option>
                        <option value="Acknowledged">Acknowledge</option>
                        <option value="Dismissed">Dismiss</option>
                      </select>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </Card>
    </AppShell>
  );
}
