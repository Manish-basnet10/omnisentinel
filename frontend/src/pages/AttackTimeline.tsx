import React, { useState, useEffect } from 'react';
import { AppShell } from '../components/layout/AppShell';
import { Card, CardHeader } from '../components/common/Card';
import { Badge } from '../components/common/Badge';
import { SkeletonTable } from '../components/common/States';
import { getForecastTimeline } from '../api/forecastApi';
import type { mockTimelineEvents } from '../data/mockForecast';
import { GitBranch, Check, Zap } from 'lucide-react';
import { cn } from '../utils/cn';

type TimelineEvent = typeof mockTimelineEvents[0];

export default function AttackTimeline() {
  const [events, setEvents] = useState<TimelineEvent[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getForecastTimeline().then(res => {
      setEvents(res);
      setLoading(false);
    });
  }, []);

  return (
    <AppShell>
      <div className="mb-4">
        <h2 className="text-xl font-bold text-text-primary tracking-tight">Attack Timeline</h2>
        <p className="text-sm text-text-muted mt-1">Chronological sequence of observed network events and predicted future actions.</p>
      </div>

      <Card>
        <CardHeader
          title="Event Timeline"
          icon={<GitBranch size={14} />}
          right={
            <div className="flex items-center gap-3 text-[10px]">
              <div className="flex items-center gap-1"><div className="w-2 h-2 rounded-full bg-safe" /><span>Observed Evidence</span></div>
              <div className="flex items-center gap-1"><div className="w-2 h-2 rounded-full bg-forecast" /><span>AI Predicted</span></div>
            </div>
          }
        />
        <div className="p-6">
          {loading ? <SkeletonTable rows={5} /> : (
            <div className="relative border-l-2 border-border-emphasis ml-4 space-y-8 pb-4">
              {events.map((ev, i) => {
                const isObserved = ev.type === 'observed';
                const color = isObserved ? 'text-safe' : 'text-forecast';
                const bg = isObserved ? 'bg-safe' : 'bg-forecast';
                
                return (
                  <div key={ev.id} className="relative pl-6">
                    {/* Timeline dot */}
                    <div className={cn('absolute -left-[9px] top-1 w-4 h-4 rounded-full flex items-center justify-center border-2 border-surface-2', bg)}>
                      {isObserved ? <Check size={10} className="text-surface-0" /> : <Zap size={10} className="text-surface-0" />}
                    </div>

                    <div className="flex flex-col sm:flex-row sm:items-start gap-4">
                      {/* Time & Stage */}
                      <div className="sm:w-32 flex-shrink-0 pt-0.5">
                        <div className="font-mono text-sm font-semibold text-text-primary mb-1">{ev.time}</div>
                        <Badge severity={isObserved ? (ev.stage === 'Benign' ? 'Safe' : 'High') : 'Forecast'} className="text-[9px]">{ev.stage}</Badge>
                      </div>

                      {/* Content */}
                      <div className={cn('flex-1 rounded-lg border p-4', isObserved ? 'bg-surface-3 border-border-subtle' : 'bg-forecast-bg/20 border-forecast-dim')}>
                        <div className="flex items-center justify-between mb-2">
                          <h4 className={cn('text-sm font-semibold', color)}>{ev.label}</h4>
                          {!isObserved && <Badge severity="Forecast" dot className="text-[9px]">Model Estimate</Badge>}
                        </div>
                        <p className="text-xs text-text-secondary leading-relaxed">{ev.detail}</p>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </Card>
    </AppShell>
  );
}
