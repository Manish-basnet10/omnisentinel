import React, { useState, useEffect, useCallback } from 'react';
import { AppShell } from '../components/layout/AppShell';
import { Card, CardHeader } from '../components/common/Card';
import { Badge } from '../components/common/Badge';
import { SkeletonTable, EmptyState } from '../components/common/States';
import { getAlerts, updateAlertStatus } from '../api/alertsApi';
import type { Alert, AlertStatus } from '../data/mockAlerts';
import { Bell, Search, Filter, ShieldAlert } from 'lucide-react';
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
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [loading, setLoading] = useState(true);
  const [filterSeverity, setFilterSeverity] = useState('');
  const [search, setSearch] = useState('');

  const load = useCallback(async () => {
    setLoading(true);
    const data = await getAlerts();
    setAlerts(data);
    setLoading(false);
  }, []);

  useEffect(() => { load(); }, [load]);

  async function handleStatusChange(id: string, status: AlertStatus) {
    await updateAlertStatus(id, status);
    setAlerts(prev => prev.map(a => a.id === id ? { ...a, status } : a));
  }

  const filtered = alerts.filter(a => {
    const matchSev = !filterSeverity || a.severity === filterSeverity;
    const q = search.toLowerCase();
    const matchQ = !q || a.title.toLowerCase().includes(q) || a.source.toLowerCase().includes(q) || a.id.toLowerCase().includes(q);
    return matchSev && matchQ;
  });

  return (
    <AppShell>
      <Card>
        <CardHeader
          title="Security Alerts"
          subtitle="Prioritized threat notifications"
          icon={<Bell size={14} />}
          right={
            <div className="flex items-center gap-2">
              <div className="relative">
                <Search size={12} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-text-disabled" />
                <input
                  value={search}
                  onChange={e => setSearch(e.target.value)}
                  placeholder="Search alerts…"
                  className="input pl-7 h-7 text-xs w-48"
                />
              </div>
              <div className="relative">
                <Filter size={12} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-text-disabled" />
                <select
                  value={filterSeverity}
                  onChange={e => setFilterSeverity(e.target.value)}
                  className="select pl-7 h-7 text-xs w-32"
                >
                  <option value="">All Severities</option>
                  <option value="Critical">Critical</option>
                  <option value="High">High</option>
                  <option value="Medium">Medium</option>
                  <option value="Low">Low</option>
                </select>
              </div>
            </div>
          }
        />
        <div className="overflow-x-auto">
          {loading ? <SkeletonTable rows={10} /> : (
            <table className="data-table">
              <thead>
                <tr>
                  <th>Time</th>
                  <th>ID</th>
                  <th>Severity</th>
                  <th>Alert Title</th>
                  <th>Source</th>
                  <th>Stage</th>
                  <th>Confidence</th>
                  <th>Status</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {filtered.length === 0 ? (
                  <tr><td colSpan={9}><EmptyState title="No alerts found" icon={<ShieldAlert size={24} />} /></td></tr>
                ) : filtered.map(alert => (
                  <tr key={alert.id} className={cn(alert.status === 'New' && alert.severity === 'Critical' ? 'bg-critical-bg/20' : '')}>
                    <td className="font-mono text-[11px] whitespace-nowrap">{alert.time}</td>
                    <td className="font-mono text-[11px] text-text-muted">{alert.id}</td>
                    <td><span className={cn('badge text-[10px]', SEVERITY_BADGE[alert.severity])}>{alert.severity}</span></td>
                    <td>
                      <div className="text-xs font-medium text-text-primary">{alert.title}</div>
                      <div className="text-[10px] text-text-muted mt-0.5 truncate max-w-[250px]" title={alert.detail}>{alert.detail}</div>
                    </td>
                    <td className="font-mono text-[11px] text-info">{alert.source}</td>
                    <td className="text-[11px] text-text-secondary">{alert.stage}</td>
                    <td className="text-[11px] font-semibold">{alert.confidence}%</td>
                    <td><span className={cn('text-xs', STATUS_COLOR[alert.status])}>{alert.status}</span></td>
                    <td>
                      <select
                        value={alert.status}
                        onChange={e => handleStatusChange(alert.id, e.target.value as AlertStatus)}
                        className="select h-6 py-0 px-2 text-[10px] w-28 bg-surface-2 border-border-subtle"
                      >
                        <option value="New">New</option>
                        <option value="Investigating">Investigating</option>
                        <option value="Acknowledged">Acknowledged</option>
                        <option value="Dismissed">Dismissed</option>
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
