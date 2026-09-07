const delay = (ms: number) => new Promise(r => setTimeout(r, ms));

// ─────────────────────────────────────────────────────────
// Model Performance + Explainability API  —  DEMO DATA
// All metrics are demonstration values from CIC-IDS2017
// ─────────────────────────────────────────────────────────

export const mockModelPerformance = [
  {
    id: 'lr',
    name: 'Logistic Regression',
    type: 'Baseline',
    binaryAuc: 0.9003,
    binaryF1:  0.6706,
    mcF1:      0.0457,
    precision: 0.72,
    recall:    0.63,
    fpr:       0.08,
    inferenceMs: 2,
    parameters: 60,
    description: 'Linear baseline — strong binary AUC but poor multiclass.',
  },
  {
    id: 'xgb',
    name: 'XGBoost',
    type: 'Gradient Boosting',
    binaryAuc: 0.8670,
    binaryF1:  0.7435,
    mcF1:      0.0667,
    precision: 0.79,
    recall:    0.71,
    fpr:       0.06,
    inferenceMs: 8,
    parameters: 120000,
    description: 'Static classifier — good binary performance, no temporal awareness.',
  },
  {
    id: 'lstm',
    name: 'LSTM World Model',
    type: 'Recurrent Neural Network',
    binaryAuc: 0.7988,
    binaryF1:  0.7102,
    mcF1:      0.0821,
    precision: 0.76,
    recall:    0.68,
    fpr: 0.07,
    inferenceMs: 24,
    parameters: 198420,
    description: 'Temporal model — comparable to GRU but slightly higher memory cost.',
  },
  {
    id: 'gru',
    name: 'GRU World Model',
    type: 'Recurrent Neural Network (Primary)',
    binaryAuc: 0.8153,
    binaryF1:  0.7520,
    mcF1:      0.0890,
    precision: 0.81,
    recall:    0.72,
    fpr: 0.05,
    inferenceMs: 18,
    parameters: 214157,
    description: 'Primary model — k-step rollout AUC 0.8153 (k=1) to 0.7725 (k=8). 43.6pt risk separation.',
    primary: true,
  },
];

export const mockConfusionMatrix = {
  labels: ['Benign', 'PortScan', 'DDoS', 'Patator', 'Bot', 'WebAtk'],
  matrix: [
    [9812, 42, 18, 5,  3,  2],
    [38,  1842, 12, 8,  2,  1],
    [22,  14, 2104, 3,  7,  0],
    [8,   11, 5,  934, 4,  3],
    [5,   3,  8,  6,  412, 2],
    [3,   4,  2,  5,  3,  287],
  ],
};

export const mockGlobalFeatureImportance = [
  { feature: 'SYN Packet Ratio',         importance: 0.187 },
  { feature: 'Port Diversity',            importance: 0.142 },
  { feature: 'Destination Count',         importance: 0.118 },
  { feature: 'Inter-arrival Time Std',    importance: 0.098 },
  { feature: 'Avg Packet Length',         importance: 0.087 },
  { feature: 'Connection Rate',           importance: 0.079 },
  { feature: 'Internal Traffic Volume',   importance: 0.072 },
  { feature: 'Flow Duration',             importance: 0.064 },
  { feature: 'FIN Ratio',                 importance: 0.058 },
  { feature: 'RST Ratio',                 importance: 0.045 },
  { feature: 'Bytes Per Packet',          importance: 0.038 },
  { feature: 'Unique Src Ports',          importance: 0.012 },
];

export const mockPredictionExplanation = {
  prediction: 'Lateral Movement',
  confidence: 81,
  positiveFactors: [
    { feature: 'Destination Diversity',   contribution: 0.28 },
    { feature: 'Connection Rate',         contribution: 0.22 },
    { feature: 'SYN Ratio',              contribution: 0.19 },
  ],
  negativeFactors: [
    { feature: 'Internal Traffic Volume', contribution: -0.14 },
    { feature: 'IAT Variance',           contribution: -0.09 },
  ],
};

// GET /model/performance
export async function getModelPerformance() {
  await delay(400);
  return mockModelPerformance;
}

// GET /model/explainability
export async function getModelExplainability() {
  await delay(350);
  return {
    globalImportance: mockGlobalFeatureImportance,
    predictionExplanation: mockPredictionExplanation,
    confusionMatrix: mockConfusionMatrix,
  };
}

// POST /reports
export async function generateReport(type: string, dateRange: { from: string; to: string }) {
  await delay(1200);
  return {
    id: `RPT-${Date.now()}`,
    type,
    dateRange,
    generatedAt: new Date().toISOString(),
    status: 'ready',
    message: `${type} generated successfully. Export options available below.`,
  };
}
