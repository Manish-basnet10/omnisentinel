import React from 'react';
import { FlaskConical } from 'lucide-react';

export function DemoBanner() {
  return (
    <div className="demo-banner" role="status" aria-label="Demo mode active">
      <FlaskConical size={12} />
      <span>DEMO MODE</span>
      <span className="text-text-disabled">·</span>
      <span>Synthetic Data — Not real network traffic or real model output</span>
    </div>
  );
}
