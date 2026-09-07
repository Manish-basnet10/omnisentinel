import React, { useRef } from 'react';
import { Database, X, Loader, CheckCircle, AlertTriangle, Upload, RefreshCw } from 'lucide-react';
import { useAnalysis, PROCESSING_STAGES } from '../../hooks/useAnalysis';
import { cn } from '../../utils/cn';

export function ActiveDatasetBanner() {
  const {
    activeDataset,
    analysis,
    isProcessing,
    processingStage,
    processingProgress,
    error,
    uploadDataset,
    clearDataset,
  } = useAnalysis();

  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (f) uploadDataset(f);
    e.target.value = '';  // reset so same file can be re-uploaded
  };

  // ── Processing state ───────────────────────────────────────────────────────
  if (isProcessing) {
    const stageIdx = PROCESSING_STAGES.indexOf(processingStage as any);
    return (
      <div className="px-4 py-2 bg-forecast-bg border-b border-forecast-dim flex items-center gap-3">
        <Loader size={13} className="animate-spin text-forecast flex-shrink-0" />
        <div className="flex-1 min-w-0">
          <div className="flex items-center justify-between text-xs mb-0.5">
            <span className="text-forecast font-medium truncate">{processingStage}…</span>
            <span className="text-text-muted font-mono ml-2 flex-shrink-0">{processingProgress}%</span>
          </div>
          <div className="flex items-center gap-0.5">
            {PROCESSING_STAGES.map((s, i) => (
              <div
                key={s}
                className={cn(
                  'h-1 flex-1 rounded-full transition-all',
                  i < stageIdx  ? 'bg-safe' :
                  i === stageIdx ? 'bg-forecast animate-pulse' :
                  'bg-border-subtle'
                )}
              />
            ))}
          </div>
        </div>
      </div>
    );
  }

  // ── Error state ────────────────────────────────────────────────────────────
  if (error) {
    return (
      <div className="px-4 py-2 bg-critical-bg border-b border-critical-dim flex items-center gap-2">
        <AlertTriangle size={13} className="text-critical flex-shrink-0" />
        <span className="text-xs text-critical flex-1 truncate">{error}</span>
        <button
          onClick={() => fileInputRef.current?.click()}
          className="text-[11px] text-critical underline hover:no-underline flex-shrink-0"
        >
          Try again
        </button>
        <input ref={fileInputRef} type="file" accept=".csv,.parquet,.pcap,.pcapng" hidden onChange={handleFileChange} />
      </div>
    );
  }

  // ── Active dataset ─────────────────────────────────────────────────────────
  if (activeDataset && analysis) {
    const riskColor =
      analysis.prediction.risk_level === 'CRITICAL' ? 'text-critical' :
      analysis.prediction.risk_level === 'HIGH'     ? 'text-warning' :
      analysis.prediction.risk_level === 'MEDIUM'   ? 'text-elevated' :
      'text-safe';

    return (
      <div className="px-4 py-1.5 bg-surface-2 border-b border-border-subtle flex items-center gap-3 flex-wrap">
        <CheckCircle size={12} className="text-safe flex-shrink-0" />

        <div className="flex items-center gap-2 min-w-0">
          <Database size={11} className="text-text-muted flex-shrink-0" />
          <span className="text-[11px] text-text-primary font-medium truncate max-w-[220px]">
            {activeDataset.filename}
          </span>
        </div>

        <div className="flex items-center gap-3 text-[11px] text-text-muted flex-wrap">
          <span>Rows: <span className="text-text-secondary">{activeDataset.row_count.toLocaleString()}</span></span>
          <span>Model: <span className="text-forecast">{activeDataset.model_mode === 'gru_60' ? 'PyTorch GRU' : 'Partial PyTorch MLP'}</span></span>
          <span>Risk: <span className={cn('font-semibold', riskColor)}>
            {analysis.prediction.current_risk.toFixed(1)}% {analysis.prediction.risk_level}
          </span></span>
        </div>

        <div className="flex items-center gap-1 ml-auto flex-shrink-0">
          <button
            onClick={() => fileInputRef.current?.click()}
            className="flex items-center gap-1 text-[11px] text-forecast hover:text-forecast-bright px-2 py-1 rounded hover:bg-forecast-bg transition-colors"
            title="Upload new dataset"
          >
            <Upload size={11} />
            New Dataset
          </button>
          <button
            onClick={clearDataset}
            className="p-1 text-text-muted hover:text-critical rounded hover:bg-critical-bg transition-colors"
            title="Clear active dataset"
          >
            <X size={12} />
          </button>
        </div>

        <input ref={fileInputRef} type="file" accept=".csv,.parquet,.pcap,.pcapng" hidden onChange={handleFileChange} />
      </div>
    );
  }

  // ── Empty state — no dataset ───────────────────────────────────────────────
  return (
    <div className="px-4 py-1.5 bg-surface-2 border-b border-border-subtle flex items-center gap-2">
      <Database size={12} className="text-text-disabled flex-shrink-0" />
      <span className="text-[11px] text-text-muted">No active dataset —</span>
      <button
        onClick={() => fileInputRef.current?.click()}
        className="text-[11px] text-forecast hover:text-forecast-bright underline hover:no-underline"
      >
        upload network traffic to begin analysis
      </button>
      <input ref={fileInputRef} type="file" accept=".csv,.parquet,.pcap,.pcapng" hidden onChange={handleFileChange} />
    </div>
  );
}
