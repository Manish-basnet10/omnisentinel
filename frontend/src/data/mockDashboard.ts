// ─────────────────────────────────────────────────────────
// Mock Dashboard Data  —  DEMO / SYNTHETIC DATA ONLY
// ─────────────────────────────────────────────────────────

export const mockDashboardSummary = {
  networkRisk: 78,
  networkRiskDelta: +12,
  networkRiskLevel: 'HIGH' as const,
  activeThreats: 12,
  suspiciousFlows: 1284,
  forecastConfidence: 91,
  protectedAssets: 248,
  lastUpdated: new Date().toISOString(),
};

export const mockForecastHero = {
  currentStage: 'Reconnaissance',
  predictedNextStage: 'Initial Access',
  possibleFollowingStage: 'Lateral Movement',
  progressionProbability: 82,
  confidenceLevel: 'High' as const,
  forecastHorizon: 5,
  timestamp: new Date().toISOString(),
};

export const mockAttackTrajectory = [
  { id: 'recon',      label: 'Reconnaissance',  status: 'observed'  as const, time: '09:21' },
  { id: 'scan',       label: 'Scanning',         status: 'observed'  as const, time: '09:28' },
  { id: 'access',     label: 'Initial Access',   status: 'current'   as const, time: '09:35' },
  { id: 'lateral',    label: 'Lateral Movement', status: 'forecast'  as const, time: null },
  { id: 'c2',         label: 'Command & Control',status: 'forecast'  as const, time: null },
  { id: 'exfil',      label: 'Exfiltration',     status: 'dormant'   as const, time: null },
];

export const mockRiskTimeSeries = [
  // historical
  { t: '08:50', risk: 20, stage: 'Benign',          type: 'historical' },
  { t: '09:00', risk: 28, stage: 'Benign',          type: 'historical' },
  { t: '09:10', risk: 35, stage: 'Reconnaissance',  type: 'historical' },
  { t: '09:20', risk: 44, stage: 'Reconnaissance',  type: 'historical' },
  { t: '09:30', risk: 57, stage: 'Scanning',        type: 'historical' },
  { t: '09:35', risk: 68, stage: 'Scanning',        type: 'historical' },
  { t: '09:38', risk: 72, stage: 'Initial Access',  type: 'now' },
  // forecast
  { t: '+1w',   risk: 78, stage: 'Initial Access',  type: 'forecast' },
  { t: '+2w',   risk: 82, stage: 'Lateral Movement',type: 'forecast' },
  { t: '+3w',   risk: 86, stage: 'Lateral Movement',type: 'forecast' },
  { t: '+4w',   risk: 89, stage: 'Command & Control',type: 'forecast' },
];

export const mockExplainability = {
  prediction: 'Initial Access',
  confidence: 82,
  factors: [
    { feature: 'SYN Packet Ratio',      contribution: 31, direction: 'positive' as const },
    { feature: 'Port Diversity',         contribution: 24, direction: 'positive' as const },
    { feature: 'Destination Count',      contribution: 18, direction: 'positive' as const },
    { feature: 'Inter-arrival Time',     contribution: 13, direction: 'negative' as const },
    { feature: 'Connection Rate',        contribution: 9,  direction: 'positive' as const },
    { feature: 'Avg Packet Size',        contribution: 5,  direction: 'negative' as const },
  ],
};

export const mockTrafficOverview = {
  '1H': Array.from({ length: 12 }, (_, i) => ({
    t: `${(9 + Math.floor(i / 4)).toString().padStart(2,'0')}:${((i % 4) * 15).toString().padStart(2,'0')}`,
    inbound:    Math.floor(Math.random() * 400 + 200),
    outbound:   Math.floor(Math.random() * 300 + 100),
    suspicious: Math.floor(Math.random() * 80  + 10),
    normal:     Math.floor(Math.random() * 350 + 150),
  })),
  '6H': Array.from({ length: 24 }, (_, i) => ({
    t: `${Math.floor(i / 4).toString().padStart(2,'0')}:${((i % 4) * 15).toString().padStart(2,'0')}`,
    inbound:    Math.floor(Math.random() * 500 + 200),
    outbound:   Math.floor(Math.random() * 400 + 100),
    suspicious: Math.floor(Math.random() * 120 + 20),
    normal:     Math.floor(Math.random() * 400 + 180),
  })),
  '24H': Array.from({ length: 24 }, (_, i) => ({
    t: `${i.toString().padStart(2,'0')}:00`,
    inbound:    Math.floor(Math.random() * 600 + 100),
    outbound:   Math.floor(Math.random() * 450 + 80),
    suspicious: Math.floor(Math.random() * 150 + 5),
    normal:     Math.floor(Math.random() * 500 + 100),
  })),
  '7D': Array.from({ length: 7 }, (_, i) => {
    const days = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'];
    return {
      t: days[i],
      inbound:    Math.floor(Math.random() * 8000 + 3000),
      outbound:   Math.floor(Math.random() * 6000 + 2000),
      suspicious: Math.floor(Math.random() * 1200 + 100),
      normal:     Math.floor(Math.random() * 7000 + 2500),
    };
  }),
};
