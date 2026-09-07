import React, { useState } from 'react';
import { Card, CardHeader } from '../common/Card';
import { BarChart2 } from 'lucide-react';
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
} from 'recharts';
import type { mockTrafficOverview } from '../../data/mockDashboard';

type Window = '1H' | '6H' | '24H' | '7D';
type TrafficPoint = typeof mockTrafficOverview['1H'][number];

const WINDOWS: Window[] = ['1H', '6H', '24H', '7D'];

export function TrafficOverview({ data }: { data: Record<Window, TrafficPoint[]> }) {
  const [window, setWindow] = useState<Window>('1H');
  const points = data[window];

  return (
    <Card>
      <CardHeader
        title="Traffic Overview"
        subtitle="Network flow volume across time"
        icon={<BarChart2 size={14} />}
        right={
          <div className="flex items-center gap-1">
            {WINDOWS.map(w => (
              <button
                key={w}
                onClick={() => setWindow(w)}
                className={`text-[11px] px-2 py-0.5 rounded transition-colors ${
                  window === w
                    ? 'bg-forecast-dim text-forecast-bright'
                    : 'text-text-muted hover:text-text-primary hover:bg-surface-3'
                }`}
              >
                {w}
              </button>
            ))}
          </div>
        }
      />
      <div className="p-4" style={{ height: 220 }}>
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={points} margin={{ top: 4, right: 8, left: -10, bottom: 0 }}>
            <defs>
              <linearGradient id="tgInbound" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%"  stopColor="#38bdf8" stopOpacity={0.2} />
                <stop offset="95%" stopColor="#38bdf8" stopOpacity={0} />
              </linearGradient>
              <linearGradient id="tgOutbound" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%"  stopColor="#a78bfa" stopOpacity={0.2} />
                <stop offset="95%" stopColor="#a78bfa" stopOpacity={0} />
              </linearGradient>
              <linearGradient id="tgSuspicious" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%"  stopColor="#ef4444" stopOpacity={0.3} />
                <stop offset="95%" stopColor="#ef4444" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="2 4" stroke="#1e2a3a" vertical={false} />
            <XAxis dataKey="t" tick={{ fill: '#64748b', fontSize: 10 }} axisLine={false} tickLine={false} />
            <YAxis tick={{ fill: '#64748b', fontSize: 10 }} axisLine={false} tickLine={false} />
            <Tooltip
              contentStyle={{ background: '#171d27', border: '1px solid #243040', borderRadius: 4, fontSize: 11 }}
              labelStyle={{ color: '#94a3b8' }}
            />
            <Legend wrapperStyle={{ fontSize: 11, color: '#64748b', paddingTop: 8 }} />
            <Area type="monotone" dataKey="inbound"    stroke="#38bdf8" strokeWidth={1.5} fill="url(#tgInbound)"    dot={false} name="Inbound" />
            <Area type="monotone" dataKey="outbound"   stroke="#a78bfa" strokeWidth={1.5} fill="url(#tgOutbound)"   dot={false} name="Outbound" />
            <Area type="monotone" dataKey="suspicious" stroke="#ef4444" strokeWidth={1.5} fill="url(#tgSuspicious)" dot={false} name="Suspicious" />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </Card>
  );
}
