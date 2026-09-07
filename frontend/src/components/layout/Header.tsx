import React, { useState, useEffect } from 'react';
import { Bell, Search } from 'lucide-react';
import { useLocation } from 'react-router-dom';
import { getDateTimeLabel } from '../../utils/formatters';
import { Badge } from '../common/Badge';

const PAGE_TITLES: Record<string, { title: string; subtitle: string }> = {
  '/dashboard':        { title: 'Network Security Overview', subtitle: 'Real-time monitoring and AI attack forecasting' },
  '/live-network':     { title: 'Live Network Monitor',      subtitle: 'Active connections and flow analysis' },
  '/traffic-analysis': { title: 'Traffic Analysis',          subtitle: 'Protocol, port, and behavior analytics' },
  '/alerts':           { title: 'Alert Management',          subtitle: 'Prioritized threat notifications' },
  '/attack-forecast':  { title: 'Attack Forecast Engine',    subtitle: 'Predicting where the current network trajectory is heading' },
  '/attack-timeline':  { title: 'Attack Timeline',           subtitle: 'Observed and predicted event sequence' },
  '/mitre-attack':     { title: 'MITRE ATT&CK',              subtitle: 'Tactic and technique mapping' },
  '/model-performance':{ title: 'Model Performance',         subtitle: 'Benchmark metrics across all models' },
  '/explainability':   { title: 'AI Explainability',         subtitle: 'Feature importance and prediction reasoning' },
  '/investigations':   { title: 'Investigations',            subtitle: 'Analyst workspace' },
  '/reports':          { title: 'Reports',                   subtitle: 'Generate and export security reports' },
  '/settings':         { title: 'Settings',                  subtitle: 'System configuration and preferences' },
};

export function Header({ alertCount = 3 }: { alertCount?: number }) {
  const location = useLocation();
  const [time, setTime] = useState(getDateTimeLabel());
  const meta = PAGE_TITLES[location.pathname] || { title: 'Project Possible', subtitle: '' };

  useEffect(() => {
    const id = setInterval(() => setTime(getDateTimeLabel()), 1000);
    return () => clearInterval(id);
  }, []);

  const hour = new Date().getHours();
  const greeting = hour < 12 ? 'Good Morning' : hour < 17 ? 'Good Afternoon' : 'Good Evening';

  return (
    <header className="flex items-center justify-between px-6 py-3 border-b border-border-subtle bg-surface-1 flex-shrink-0">
      <div>
        <div className="text-[11px] text-text-muted mb-0.5">{greeting}, Security Analyst</div>
        <h1 className="text-base font-semibold text-text-primary tracking-tight">{meta.title}</h1>
        {meta.subtitle && <p className="text-[11px] text-text-muted">{meta.subtitle}</p>}
      </div>

      <div className="flex items-center gap-4">
        {/* Search */}
        <div className="relative hidden md:block">
          <Search size={13} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-text-disabled" />
          <input
            type="search"
            placeholder="Search flows, alerts, hosts…"
            className="input pl-8 w-56 h-8 text-xs"
            aria-label="Search"
          />
        </div>

        {/* Live status */}
        <div className="flex items-center gap-1.5 text-[11px] text-safe">
          <span className="status-dot status-online status-pulse" />
          <span>LIVE</span>
        </div>

        {/* Time */}
        <div className="font-mono text-[11px] text-text-muted hidden lg:block">
          {time}
        </div>

        {/* Alerts bell */}
        <button className="relative btn-ghost btn btn-xs p-1.5" aria-label={`${alertCount} unread alerts`}>
          <Bell size={15} />
          {alertCount > 0 && (
            <span className="absolute -top-0.5 -right-0.5 w-4 h-4 rounded-full bg-critical text-white text-[9px] flex items-center justify-center font-bold">
              {alertCount > 9 ? '9+' : alertCount}
            </span>
          )}
        </button>
      </div>
    </header>
  );
}
