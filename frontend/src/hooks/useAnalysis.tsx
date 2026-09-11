import React, { createContext, useContext, useState, useCallback, useEffect, useRef } from 'react';
import client from '../api/client';

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────
export interface DatasetMeta {
  _id:           string;
  filename:      string;
  format:        string;
  row_count:     number;
  feature_count: number;
  model_mode:    string;
  created_at:    string;
}

export interface AnalysisSession {
  dataset_id:  string;
  created_at:  string;
  model: {
    type:          string;
    mode:          string;
    features_used: number;
  };
  dataset: {
    rows:                number;
    features_available:  number;
    features_used:       number;
    features_engineered: number;
    missing_features:    string[];
    format:              string;
    filename:            string;
  };
  prediction: {
    current_risk:         number;
    risk_level:           string;
    risk_trend:           string;
    attack_probability:   number;
    predicted_next_stage: string | null;
    predicted_technique:  string | null;
    confidence:           number | null;
  };
  forecast:            any[];
  mitre_progression:   any[];
  timeline:            any[];
  alerts:              any[];
  investigations:      any[];
  top_saliency_features: any[];
  traffic_analysis:    any;
  model_metrics:       any;
}

export interface AnalysisContextValue {
  datasetId:          string | null;
  activeDataset:      DatasetMeta | null;
  analysis:           AnalysisSession | null;
  isProcessing:       boolean;
  processingStage:    string;
  processingProgress: number;
  error:              string | null;
  uploadDataset:      (file: File) => Promise<void>;
  refreshAnalysis:    () => Promise<void>;
  clearDataset:       () => void;
}

// ─────────────────────────────────────────────────────────────────────────────
// Processing stages (shown to user during upload)
// ─────────────────────────────────────────────────────────────────────────────
export const PROCESSING_STAGES = [
  'Uploading',
  'Feature Engineering',
  'Running PyTorch Model',
  'Generating Forecast',
  'Generating MITRE Analysis',
  'Saving Analysis',
  'Complete',
] as const;

const STAGE_PROGRESS: Record<string, number> = {
  'Uploading':              10,
  'Feature Engineering':    35,
  'Running PyTorch Model':  55,
  'Generating Forecast':    70,
  'Generating MITRE Analysis': 82,
  'Saving Analysis':        92,
  'Complete':               100,
};

// ─────────────────────────────────────────────────────────────────────────────
// Context
// ─────────────────────────────────────────────────────────────────────────────
const AnalysisContext = createContext<AnalysisContextValue | null>(null);

const LS_DATASET_KEY = 'omnisentinel_dataset_id';

// ─────────────────────────────────────────────────────────────────────────────
// Provider
// ─────────────────────────────────────────────────────────────────────────────
export function AnalysisProvider({ children }: { children: React.ReactNode }) {
  const [datasetId, setDatasetId]           = useState<string | null>(() => localStorage.getItem(LS_DATASET_KEY));
  const [activeDataset, setActiveDataset]   = useState<DatasetMeta | null>(null);
  const [analysis, setAnalysis]             = useState<AnalysisSession | null>(null);
  const [isProcessing, setIsProcessing]     = useState(false);
  const [processingStage, setProcessingStage] = useState('');
  const [processingProgress, setProcessingProgress] = useState(0);
  const [error, setError]                   = useState<string | null>(null);
  const abortRef                            = useRef<AbortController | null>(null);

  // ── Simulate progress stages during upload/processing ────────────────────
  const simulateProgress = useCallback((onComplete?: () => void) => {
    const stages = Object.keys(STAGE_PROGRESS);
    let idx = 0;

    const advance = () => {
      if (idx >= stages.length) {
        onComplete?.();
        return;
      }
      const stage = stages[idx];
      setProcessingStage(stage);
      setProcessingProgress(STAGE_PROGRESS[stage]);
      idx++;
      if (stage !== 'Complete') {
        // Advance slower for ML stages, faster for others
        const delay = stage === 'Running PyTorch Model' ? 3000 : stage === 'Feature Engineering' ? 1500 : 800;
        setTimeout(advance, delay);
      }
    };

    advance();
  }, []);

  // ── Fetch analysis from backend ──────────────────────────────────────────
  const fetchAnalysis = useCallback(async (id: string) => {
    try {
      const [datasetRes, analysisRes] = await Promise.all([
        client.get(`/api/datasets/${id}`),
        client.get(`/api/datasets/${id}/analysis`),
      ]);
      setActiveDataset(datasetRes.data);
      setAnalysis(analysisRes.data);
      setError(null);
    } catch (err: any) {
      if (err?.response?.status === 404) {
        // Dataset was deleted or belongs to another user — clear it
        localStorage.removeItem(LS_DATASET_KEY);
        setDatasetId(null);
        setActiveDataset(null);
        setAnalysis(null);
      } else {
        setError(err?.response?.data?.detail || 'Failed to load analysis');
      }
    }
  }, []);

  // ── On app start: restore dataset from localStorage or bootstrap default ──────
  useEffect(() => {
    const initializeDataset = async () => {
      try {
        if (datasetId) {
          await fetchAnalysis(datasetId);
        } else {
          // Check for active dataset in backend
          const activeRes = await client.get('/api/datasets/active');
          if (activeRes.data.dataset_id) {
            localStorage.setItem(LS_DATASET_KEY, activeRes.data.dataset_id);
            setDatasetId(activeRes.data.dataset_id);
            await fetchAnalysis(activeRes.data.dataset_id);
          } else if (activeRes.data.needs_bootstrap) {
            // Bootstrap default SIH dataset
            setIsProcessing(true);
            setProcessingStage('Generating Initial Analysis from SIH Dataset');
            setProcessingProgress(50);
            
            const bootstrapRes = await client.post('/api/datasets/bootstrap');
            if (bootstrapRes.data.dataset_id) {
              localStorage.setItem(LS_DATASET_KEY, bootstrapRes.data.dataset_id);
              setDatasetId(bootstrapRes.data.dataset_id);
              await fetchAnalysis(bootstrapRes.data.dataset_id);
            }
            
            setProcessingStage('Complete');
            setProcessingProgress(100);
            setIsProcessing(false);
          }
        }
      } catch (err) {
        console.error("Failed to initialize dataset:", err);
        setIsProcessing(false);
      }
    };

    initializeDataset();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);   // intentionally only on mount

  // ── Upload a new dataset ─────────────────────────────────────────────────
  const uploadDataset = useCallback(async (file: File) => {
    setIsProcessing(true);
    setError(null);
    setProcessingStage('Uploading');
    setProcessingProgress(5);

    // Abort any in-flight requests
    if (abortRef.current) abortRef.current.abort();
    abortRef.current = new AbortController();

    // Kick off progress simulation immediately so UI feels responsive
    simulateProgress();

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await client.post('/api/datasets/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        signal: abortRef.current.signal,
      });

      const { dataset_id } = response.data;

      // Persist to localStorage so browser refresh recovers it
      localStorage.setItem(LS_DATASET_KEY, dataset_id);
      setDatasetId(dataset_id);

      // Fetch full analysis (pipeline already ran on backend)
      await fetchAnalysis(dataset_id);

      setProcessingStage('Complete');
      setProcessingProgress(100);
    } catch (err: any) {
      if (err?.name === 'CanceledError') return;  // aborted
      const msg = err?.response?.data?.detail || err?.message || 'Upload failed';
      setError(msg);
      setProcessingStage('');
      setProcessingProgress(0);
    } finally {
      setIsProcessing(false);
    }
  }, [fetchAnalysis, simulateProgress]);

  // ── Refresh analysis (re-fetch, no re-upload) ────────────────────────────
  const refreshAnalysis = useCallback(async () => {
    if (!datasetId) return;
    await fetchAnalysis(datasetId);
  }, [datasetId, fetchAnalysis]);

  // ── Clear active dataset ─────────────────────────────────────────────────
  const clearDataset = useCallback(() => {
    localStorage.removeItem(LS_DATASET_KEY);
    setDatasetId(null);
    setActiveDataset(null);
    setAnalysis(null);
    setError(null);
    setProcessingStage('');
    setProcessingProgress(0);
  }, []);

  return (
    <AnalysisContext.Provider value={{
      datasetId,
      activeDataset,
      analysis,
      isProcessing,
      processingStage,
      processingProgress,
      error,
      uploadDataset,
      refreshAnalysis,
      clearDataset,
    }}>
      {children}
    </AnalysisContext.Provider>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Hook
// ─────────────────────────────────────────────────────────────────────────────
export function useAnalysis(): AnalysisContextValue {
  const ctx = useContext(AnalysisContext);
  if (!ctx) throw new Error('useAnalysis must be used within AnalysisProvider');
  return ctx;
}
