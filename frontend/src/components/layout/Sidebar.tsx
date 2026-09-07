import React, { useState } from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import {
  LayoutDashboard, Radio, BarChart3, Bell, Zap, GitBranch, Shield,
  Brain, LineChart, Search, FileText, Settings, ChevronLeft, ChevronRight,
  User, Cpu, Wifi, Globe, LogOut,
} from 'lucide-react';
import { useAuth } from '../../hooks/useAuth';
import { cn } from '../../utils/cn';

interface NavItem { label: string; to: string; icon: React.ReactNode; }
interface NavGroup { section: string; items: NavItem[]; }

const NAV: NavGroup[] = [
  {
    section: '',
    items: [{ label: 'Overview', to: '/dashboard', icon: <LayoutDashboard size={15} /> }],
  },
  {
    section: 'Monitor',
    items: [
      { label: 'Live Network',      to: '/live-network',     icon: <Radio size={15} /> },
      { label: 'Traffic Analysis',  to: '/traffic-analysis', icon: <BarChart3 size={15} /> },
      { label: 'Alerts',            to: '/alerts',           icon: <Bell size={15} /> },
    ],
  },
  {
    section: 'Intelligence',
    items: [
      { label: 'Attack Forecast',   to: '/attack-forecast',  icon: <Zap size={15} /> },
      { label: 'Attack Timeline',   to: '/attack-timeline',  icon: <GitBranch size={15} /> },
      { label: 'MITRE ATT&CK',      to: '/mitre-attack',     icon: <Shield size={15} /> },
    ],
  },
  {
    section: 'AI',
    items: [
      { label: 'Model Performance', to: '/model-performance',icon: <Brain size={15} /> },
      { label: 'Explainability',    to: '/explainability',   icon: <LineChart size={15} /> },
    ],
  },
  {
    section: 'Operations',
    items: [
      { label: 'Investigations',    to: '/investigations',   icon: <Search size={15} /> },
      { label: 'Reports',           to: '/reports',          icon: <FileText size={15} /> },
    ],
  },
  {
    section: 'System',
    items: [
      { label: 'Settings',          to: '/settings',         icon: <Settings size={15} /> },
    ],
  },
];

export function Sidebar() {
  const [collapsed, setCollapsed] = useState(false);
  const { user, logout } = useAuth();
  const location = useLocation();

  return (
    <aside className={cn('sidebar flex-shrink-0 flex flex-col transition-all duration-200 z-20', collapsed && 'collapsed')}>
      {/* Brand */}
      <div className="px-3 py-4 border-b border-border-subtle flex items-center justify-between">
        {!collapsed && (
          <div>
            <div className="text-[13px] font-bold tracking-wider text-text-primary">PROJECT POSSIBLE</div>
            <div className="text-[10px] text-forecast tracking-widest uppercase mt-0.5">AI Network Defence</div>
          </div>
        )}
        {collapsed && (
          <div className="w-6 h-6 rounded bg-forecast-dim flex items-center justify-center text-forecast-bright text-[10px] font-bold mx-auto">PP</div>
        )}
        <button
          onClick={() => setCollapsed(c => !c)}
          className="btn-ghost btn btn-xs ml-auto"
          aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
        >
          {collapsed ? <ChevronRight size={14} /> : <ChevronLeft size={14} />}
        </button>
      </div>

      {/* Navigation */}
      <nav className="flex-1 overflow-y-auto py-2 space-y-0.5" role="navigation" aria-label="Main navigation">
        {NAV.map((group) => (
          <div key={group.section}>
            {group.section && !collapsed && (
              <div className="sidebar-section">{group.section}</div>
            )}
            {group.section && collapsed && <div className="h-3" />}
            {group.items.map((item) => {
              const isActive = location.pathname === item.to ||
                (item.to !== '/dashboard' && location.pathname.startsWith(item.to));
              return (
                <NavLink
                  key={item.to}
                  to={item.to}
                  className={cn(
                    'sidebar-item mx-1',
                    isActive && 'active',
                    collapsed && 'justify-center px-0'
                  )}
                  title={collapsed ? item.label : undefined}
                  aria-label={item.label}
                  aria-current={isActive ? 'page' : undefined}
                >
                  <span className="sidebar-icon flex-shrink-0">{item.icon}</span>
                  {!collapsed && <span className="text-[13px]">{item.label}</span>}
                </NavLink>
              );
            })}
          </div>
        ))}
      </nav>

      {/* Footer status */}
      {!collapsed && (
        <div className="border-t border-border-subtle px-3 py-3 space-y-1.5">
          <StatusRow icon={<Cpu size={11} />} label="AI Engine" status="online" />
          <StatusRow icon={<Wifi size={11} />} label="Network Monitor" status="online" />
          <StatusRow icon={<Globe size={11} />} label="API" status="online" />
        </div>
      )}

      {/* User profile */}
      <div className={cn(
        'border-t border-border-subtle px-3 py-3 flex items-center gap-2',
        collapsed && 'justify-center'
      )}>
        <div className="w-7 h-7 rounded-full bg-forecast-dim flex items-center justify-center flex-shrink-0">
          <User size={13} className="text-forecast-bright" />
        </div>
        {!collapsed && (
          <>
            <div className="flex-1 min-w-0">
              <div className="text-[12px] font-medium text-text-primary truncate">{user?.name}</div>
              <div className="text-[10px] text-text-muted">{user?.team}</div>
            </div>
            <button
              onClick={logout}
              className="btn-ghost btn btn-xs p-1"
              title="Sign out"
              aria-label="Sign out"
            >
              <LogOut size={13} />
            </button>
          </>
        )}
      </div>
    </aside>
  );
}

function StatusRow({ icon, label, status }: { icon: React.ReactNode; label: string; status: 'online' | 'offline' | 'warning' }) {
  return (
    <div className="flex items-center gap-2 text-[11px] text-text-muted">
      <span>{icon}</span>
      <span className="flex-1">{label}</span>
      <span className={cn('status-dot', status === 'online' ? 'status-online status-pulse' : status === 'warning' ? 'status-warning' : 'status-error')} />
      <span className={cn('text-[10px]', status === 'online' ? 'text-safe' : 'text-critical')}>
        {status === 'online' ? 'Online' : status === 'warning' ? 'Degraded' : 'Offline'}
      </span>
    </div>
  );
}
