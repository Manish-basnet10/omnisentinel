import React, { useState, useEffect, useCallback } from 'react';
import { AppShell } from '../components/layout/AppShell';
import { Card, CardHeader } from '../components/common/Card';
import { Badge } from '../components/common/Badge';
import { SkeletonTable, EmptyState, ProgressBar } from '../components/common/States';
import { formatBytes, formatNumber, formatDuration } from '../utils/formatters';
import { getLiveNetworkStats, getNetworkFlows, subscribeToLiveFlows } from '../api/networkApi';
import { mockFlows, type NetworkFlow } from '../data/mockTraffic';
import { Radio, Search, RefreshCw, ChevronLeft, ChevronRight } from 'lucide-react';
import { cn } from '../utils/cn';

const RISK_BADGE: Record<string, string> = {
  Critical: 'badge-critical', High: 'badge-warning',
  Medium: 'badge-elevated', Low: 'badge-info', Safe: 'badge-safe',
};
const STATUS_COLOR: Record<string, string> = {
  Active: 'text-info', Blocked: 'text-critical',
  Suspicious: 'text-warning', Normal: 'text-safe',
};

export default function LiveNetwork() {
  const [stats, setStats]       = useState<any>(null);
  const [flows, setFlows]       = useState<NetworkFlow[]>([]);
  const [loading, setLoading]   = useState(true);
  const [page, setPage]         = useState(1);
  const [search, setSearch]     = useState('');
  const [sortKey, setSortKey]   = useState<keyof NetworkFlow>('timestamp');
  const [sortDir, setSortDir]   = useState<'asc' | 'desc'>('desc');
  const [filterRisk, setFilterRisk] = useState('');
  const PAGE_SIZE = 15;

  const load = useCallback(async () => {
    const [s, f] = await Promise.all([getLiveNetworkStats(), getNetworkFlows(1, 80)]);
    setStats(s); setFlows(f.flows); setLoading(false);
  }, []);

  useEffect(() => {
    load();
    const unsubscribe = subscribeToLiveFlows((newFlows) => {
      setFlows(prev => [...newFlows, ...prev].slice(0, 100));
      setStats((prev: any) => prev ? { ...prev, activeFlows: prev.activeFlows + Math.floor(Math.random() * 5) } : prev);
    });
    return unsubscribe;
  }, [load]);

  // Filter + sort
  const filtered = flows.filter(f => {
    const q = search.toLowerCase();
    const matchSearch = !q || f.source.includes(q) || f.destination.includes(q) || f.protocol.toLowerCase().includes(q);
    const matchRisk = !filterRisk || f.risk === filterRisk;
    return matchSearch && matchRisk;
  }).sort((a, b) => {
    const av = a[sortKey] as any, bv = b[sortKey] as any;
    return sortDir === 'asc' ? (av < bv ? -1 : 1) : (av > bv ? -1 : 1);
  });

  const paginated = filtered.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE);
  const totalPages = Math.ceil(filtered.length / PAGE_SIZE);

  function handleSort(key: keyof NetworkFlow) {
    if (sortKey === key) setSortDir(d => d === 'asc' ? 'desc' : 'asc');
    else { setSortKey(key); setSortDir('desc'); }
  }

  return (
    <AppShell alertCount={stats?.suspiciousFlows ? Math.floor(stats.suspiciousFlows / 10) : 3}>
      {/* Stats row */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 mb-4">
        {[
          { label: 'Total Connections', value: stats ? formatNumber(stats.totalConnections) : '—', color: 'text-text-primary' },
          { label: 'Active Flows',      value: stats ? formatNumber(stats.activeFlows) : '—',      color: 'text-info' },
          { label: 'Suspicious Flows',  value: stats ? formatNumber(stats.suspiciousFlows) : '—',  color: 'text-warning' },
          { label: 'Blocked',           value: stats ? formatNumber(stats.blockedConnections) : '—',color: 'text-critical' },
        ].map(s => (
          <div key={s.label} className="card p-4 flex items-center justify-between">
            <div>
              <div className={`text-2xl font-bold ${s.color}`}>{s.value}</div>
              <div className="text-[11px] text-text-muted">{s.label}</div>
            </div>
            {s.label === 'Active Flows' && (
              <div className="flex items-center gap-1.5 text-[11px] text-safe">
                <span className="status-dot status-online status-pulse" />
                LIVE
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Flow table */}
      <Card>
        <CardHeader
          title="Live Traffic Flows"
          icon={<Radio size={14} />}
          right={
            <div className="flex items-center gap-2">
              {/* Search */}
              <div className="relative">
                <Search size={12} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-text-disabled" />
                <input
                  value={search}
                  onChange={e => { setSearch(e.target.value); setPage(1); }}
                  placeholder="Search source, dest, protocol…"
                  className="input pl-7 h-7 text-xs w-48"
                />
              </div>
              {/* Risk filter */}
              <select
                value={filterRisk}
                onChange={e => { setFilterRisk(e.target.value); setPage(1); }}
                className="select h-7 text-xs w-28"
              >
                <option value="">All Risk</option>
                {['Critical','High','Medium','Low','Safe'].map(r => <option key={r} value={r}>{r}</option>)}
              </select>
              <button onClick={load} className="btn btn-ghost btn-xs p-1.5" aria-label="Refresh">
                <RefreshCw size={13} />
              </button>
            </div>
          }
        />
        <div className="overflow-x-auto">
          {loading ? <SkeletonTable rows={8} /> : (
            <table className="data-table">
              <thead>
                <tr>
                  {[
                    { key: 'timestamp', label: 'Timestamp' },
                    { key: 'source',    label: 'Source' },
                    { key: 'destination', label: 'Destination' },
                    { key: 'protocol',  label: 'Protocol' },
                    { key: 'port',      label: 'Port' },
                    { key: 'packets',   label: 'Packets' },
                    { key: 'bytes',     label: 'Bytes' },
                    { key: 'risk',      label: 'Risk' },
                    { key: 'status',    label: 'Status' },
                  ].map(col => (
                    <th
                      key={col.key}
                      onClick={() => handleSort(col.key as keyof NetworkFlow)}
                      className="cursor-pointer select-none hover:text-text-primary transition-colors"
                    >
                      {col.label}{sortKey === col.key ? (sortDir === 'asc' ? ' ↑' : ' ↓') : ''}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {paginated.length === 0 ? (
                  <tr><td colSpan={9}><EmptyState title="No flows match your filter" /></td></tr>
                ) : paginated.map(flow => (
                  <tr key={flow.id} className={cn(flow.risk === 'Critical' && 'bg-critical-bg/30', flow.risk === 'High' && 'bg-warning-bg/20')}>
                    <td className="font-mono text-[11px]">{flow.timestamp}</td>
                    <td className="font-mono text-[11px] text-info">{flow.source}</td>
                    <td className="font-mono text-[11px]">{flow.destination}</td>
                    <td><span className="badge badge-muted">{flow.protocol}</span></td>
                    <td className="font-mono text-[11px]">{flow.port}</td>
                    <td className="text-[11px]">{formatNumber(flow.packets)}</td>
                    <td className="text-[11px]">{formatBytes(flow.bytes)}</td>
                    <td><span className={cn('badge text-[10px]', RISK_BADGE[flow.risk])}>{flow.risk}</span></td>
                    <td><span className={cn('text-xs font-medium', STATUS_COLOR[flow.status])}>{flow.status}</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
        {/* Pagination */}
        {!loading && totalPages > 1 && (
          <div className="flex items-center justify-between px-4 py-2 border-t border-border-subtle text-xs text-text-muted">
            <span>Showing {(page - 1) * PAGE_SIZE + 1}–{Math.min(page * PAGE_SIZE, filtered.length)} of {filtered.length}</span>
            <div className="flex items-center gap-1">
              <button onClick={() => setPage(p => Math.max(1, p - 1))} disabled={page === 1} className="btn btn-ghost btn-xs p-1">
                <ChevronLeft size={13} />
              </button>
              <span>{page} / {totalPages}</span>
              <button onClick={() => setPage(p => Math.min(totalPages, p + 1))} disabled={page === totalPages} className="btn btn-ghost btn-xs p-1">
                <ChevronRight size={13} />
              </button>
            </div>
          </div>
        )}
      </Card>
    </AppShell>
  );
}
