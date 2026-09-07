import React from 'react';
import { AppShell } from '../components/layout/AppShell';
import { Card, CardHeader } from '../components/common/Card';
import { Settings as SettingsIcon, Shield, Zap, Globe, Save } from 'lucide-react';

export default function Settings() {
  return (
    <AppShell>
      <div className="mb-6 flex justify-between items-start">
        <div>
          <h2 className="text-xl font-bold text-text-primary tracking-tight">System Settings</h2>
          <p className="text-sm text-text-muted mt-1">Configure OmniSentinel UI and AI Engine parameters.</p>
        </div>
        <button className="btn btn-primary">
          <Save size={14} /> Save Changes
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader title="AI Forecast Configuration" icon={<Zap size={14} />} />
          <div className="p-5 space-y-4">
            <div>
              <label className="block text-xs text-text-secondary mb-1.5">Primary World Model</label>
              <select className="select w-full" defaultValue="gru">
                <option value="gru">GRU World Model (Phase 8) - Recommended</option>
                <option value="lstm">LSTM World Model (Phase 9)</option>
                <option value="xgb" disabled>XGBoost (Static Baseline)</option>
              </select>
            </div>
            
            <div>
              <label className="block text-xs text-text-secondary mb-1.5">Autoregressive Rollout Horizon (K-steps)</label>
              <input type="range" min="1" max="10" defaultValue="8" className="w-full accent-forecast" />
              <div className="flex justify-between text-[10px] text-text-muted mt-1">
                <span>1 window</span>
                <span>8 windows (Default)</span>
                <span>10 windows</span>
              </div>
            </div>

            <div>
              <label className="block text-xs text-text-secondary mb-1.5">Risk Score Discount Factor (γ)</label>
              <input type="number" step="0.05" min="0" max="1" defaultValue="0.85" className="input font-mono" />
              <p className="text-[10px] text-text-muted mt-1">Determines how much future predicted risk influences current risk score.</p>
            </div>
          </div>
        </Card>

        <Card>
          <CardHeader title="Alerting & Thresholds" icon={<Shield size={14} />} />
          <div className="p-5 space-y-4">
            <div>
              <label className="block text-xs text-text-secondary mb-1.5">Critical Alert Risk Threshold</label>
              <input type="number" min="0" max="100" defaultValue="80" className="input font-mono" />
            </div>
            <div>
              <label className="block text-xs text-text-secondary mb-1.5">High Alert Risk Threshold</label>
              <input type="number" min="0" max="100" defaultValue="65" className="input font-mono" />
            </div>
            
            <div className="pt-4 border-t border-border-subtle">
              <label className="flex items-center gap-3 cursor-pointer">
                <input type="checkbox" defaultChecked className="w-4 h-4 rounded bg-surface-3 border-border-default accent-warning" />
                <div>
                  <div className="text-sm text-text-primary">Predictive Alerting</div>
                  <div className="text-[10px] text-text-muted">Generate alerts based on AI forecast probability before attack occurs.</div>
                </div>
              </label>
            </div>
          </div>
        </Card>

        <Card>
          <CardHeader title="API Connection" icon={<Globe size={14} />} />
          <div className="p-5 space-y-4">
            <div>
              <label className="block text-xs text-text-secondary mb-1.5">Inference Server URL</label>
              <input type="text" defaultValue={import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'} className="input font-mono" disabled />
              <p className="text-[10px] text-text-muted mt-1">Configured via VITE_API_BASE_URL</p>
            </div>
            
            <div className="flex items-center justify-between p-3 bg-surface-3 rounded border border-border-subtle">
              <div className="text-xs text-text-primary">Connection Status</div>
              <div className="flex items-center gap-1.5 text-[11px] text-safe font-medium">
                <span className="w-2 h-2 rounded-full bg-safe animate-pulse" />
                Connected
              </div>
            </div>
          </div>
        </Card>
      </div>
    </AppShell>
  );
}
