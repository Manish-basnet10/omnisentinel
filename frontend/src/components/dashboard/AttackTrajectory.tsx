import React from 'react';
import { Check, Clock } from 'lucide-react';
import { Card, CardHeader } from '../common/Card';
import { Badge } from '../common/Badge';
import { cn } from '../../utils/cn';
import type { mockAttackTrajectory } from '../../data/mockDashboard';

type TrajectoryStage = typeof mockAttackTrajectory[number];
type Status = TrajectoryStage['status'];

const nodeStyle: Record<Status, { circle: string; label: string; connector: string }> = {
  observed:  { circle: 'stage-observed',  label: 'text-safe',             connector: 'bg-safe' },
  current:   { circle: 'stage-current',   label: 'text-elevated font-bold', connector: 'bg-elevated' },
  forecast:  { circle: 'stage-forecast',  label: 'text-forecast',  connector: 'bg-forecast' },
  dormant:   { circle: 'stage-dormant',   label: 'text-text-disabled',     connector: 'bg-border-subtle' },
};

const nodeIcon: Record<Status, React.ReactNode> = {
  observed: <Check size={12} />,
  current:  <span className="text-[10px] font-bold">NOW</span>,
  forecast: <span className="text-[10px]">◆</span>,
  dormant:  <span className="text-[10px]">○</span>,
};

export function AttackTrajectory({ stages }: { stages: TrajectoryStage[] }) {
  return (
    <Card>
      <CardHeader
        title="Attack Progression Trajectory"
        subtitle="Observed stages → Current → Forecast"
        right={
          <div className="flex items-center gap-3 text-[10px]">
            <LegendItem color="bg-safe"             label="Observed" />
            <LegendItem color="bg-elevated"         label="Current" />
            <LegendItem color="bg-forecast" label="Forecast" />
            <LegendItem color="bg-border-subtle"    label="Not observed" />
          </div>
        }
      />
      <div className="p-4">
        {/* Timeline row */}
        <div className="flex items-start relative">
          {stages.map((stage, i) => {
            const style = nodeStyle[stage.status];
            const isLast = i === stages.length - 1;
            return (
              <React.Fragment key={stage.id}>
                <div className="stage-node flex-1 min-w-0">
                  {/* Label above */}
                  <div className="text-center mb-2">
                    {stage.time && (
                      <div className="flex items-center justify-center gap-1 text-[10px] text-text-muted mb-0.5">
                        <Clock size={9} />
                        {stage.time}
                      </div>
                    )}
                    {!stage.time && stage.status === 'forecast' && (
                      <div className="text-[10px] text-forecast mb-0.5">Predicted</div>
                    )}
                    {!stage.time && stage.status === 'dormant' && (
                      <div className="text-[10px] text-text-disabled mb-0.5">&nbsp;</div>
                    )}
                  </div>

                  {/* Circle */}
                  <div className="flex items-center justify-center">
                    <div className={cn(style.circle, 'flex items-center justify-center')}>
                      {nodeIcon[stage.status]}
                    </div>
                  </div>

                  {/* Label below */}
                  <div className={cn('text-center mt-2 text-[11px] px-1', style.label)}>
                    {stage.label}
                  </div>

                  {/* Status badge */}
                  <div className="flex justify-center mt-1">
                    {stage.status === 'observed' && <Badge severity="Safe"     className="text-[9px] py-0 px-1">Confirmed</Badge>}
                    {stage.status === 'current'  && <Badge severity="Medium"   className="text-[9px] py-0 px-1">Active</Badge>}
                    {stage.status === 'forecast' && <Badge severity="Forecast" className="text-[9px] py-0 px-1">Forecast</Badge>}
                    {stage.status === 'dormant'  && <Badge severity="Muted"    className="text-[9px] py-0 px-1">Not observed</Badge>}
                  </div>
                </div>

                {/* Connector line */}
                {!isLast && (
                  <div className="flex items-center justify-center mt-8 flex-shrink-0 w-4">
                    <div className={cn('h-0.5 w-full', style.connector, 'opacity-60')} />
                  </div>
                )}
              </React.Fragment>
            );
          })}
        </div>

        {/* Axis labels */}
        <div className="flex justify-between mt-3 text-[10px] text-text-disabled border-t border-border-subtle pt-2">
          <span>← Past (Confirmed Evidence)</span>
          <span>Present</span>
          <span>AI Forecast (Model Estimates) →</span>
        </div>
      </div>
    </Card>
  );
}

function LegendItem({ color, label }: { color: string; label: string }) {
  return (
    <div className="flex items-center gap-1">
      <div className={cn('w-2.5 h-2.5 rounded-full', color)} />
      <span className="text-text-muted">{label}</span>
    </div>
  );
}
