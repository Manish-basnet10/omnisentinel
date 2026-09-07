// ─────────────────────────────────────────────────────────
// Mock Forecast Data  —  DEMO / SYNTHETIC DATA ONLY
// All model metrics and probabilities are demonstration values
// ─────────────────────────────────────────────────────────

export type AttackStage =
  | 'Benign'
  | 'Reconnaissance'
  | 'Scanning'
  | 'Discovery'
  | 'Initial Access'
  | 'Credential Access'
  | 'Lateral Movement'
  | 'Privilege Escalation'
  | 'Command & Control'
  | 'Exfiltration'
  | 'Impact';

export interface ForecastStep {
  horizon: string;       // "Now", "+1", "+2", etc.
  stage: AttackStage;
  probability: number;   // 0–100
  confidence: number;    // 0–100
  riskScore: number;     // 0–100
  trafficPattern: string;
  description: string;
}

export interface ForwardSimulation {
  steps: ForecastStep[];
  overallRisk: number;
  horizon: number;
  runAt: string;
}

export const mockForecastSteps: Record<number, ForecastStep[]> = {
  1: [
    {
      horizon: 'Now',
      stage: 'Reconnaissance',
      probability: 72,
      confidence: 88,
      riskScore: 68,
      trafficPattern: 'High SYN ratio, port scan signatures',
      description: 'Observed behavior consistent with active network reconnaissance.',
    },
    {
      horizon: '+1',
      stage: 'Initial Access',
      probability: 82,
      confidence: 91,
      riskScore: 78,
      trafficPattern: 'Credential probing, auth failures',
      description: 'Model estimates high probability of transition to initial access attempts.',
    },
  ],
  3: [
    {
      horizon: 'Now',
      stage: 'Reconnaissance',
      probability: 72,
      confidence: 88,
      riskScore: 68,
      trafficPattern: 'High SYN ratio, port scan signatures',
      description: 'Observed behavior consistent with active network reconnaissance.',
    },
    {
      horizon: '+1',
      stage: 'Scanning',
      probability: 79,
      confidence: 86,
      riskScore: 73,
      trafficPattern: 'Service enumeration, banner grabbing',
      description: 'Predicted escalation to systematic service scanning.',
    },
    {
      horizon: '+2',
      stage: 'Initial Access',
      probability: 82,
      confidence: 81,
      riskScore: 78,
      trafficPattern: 'Credential probing, auth failures',
      description: 'Estimated progression to access attempts.',
    },
    {
      horizon: '+3',
      stage: 'Lateral Movement',
      probability: 61,
      confidence: 71,
      riskScore: 83,
      trafficPattern: 'Internal host pivoting, SMB traffic',
      description: 'If initial access succeeds, lateral movement is a probable next step.',
    },
  ],
  5: [
    {
      horizon: 'Now',
      stage: 'Reconnaissance',
      probability: 72,
      confidence: 88,
      riskScore: 68,
      trafficPattern: 'High SYN ratio, port scan signatures',
      description: 'Observed behavior consistent with active network reconnaissance.',
    },
    {
      horizon: '+1',
      stage: 'Scanning',
      probability: 79,
      confidence: 86,
      riskScore: 73,
      trafficPattern: 'Service enumeration, banner grabbing',
      description: 'Predicted escalation to systematic service scanning.',
    },
    {
      horizon: '+2',
      stage: 'Initial Access',
      probability: 82,
      confidence: 81,
      riskScore: 78,
      trafficPattern: 'Credential probing, auth failures',
      description: 'Estimated progression to access attempts.',
    },
    {
      horizon: '+3',
      stage: 'Lateral Movement',
      probability: 61,
      confidence: 71,
      riskScore: 83,
      trafficPattern: 'Internal host pivoting, SMB traffic',
      description: 'If initial access succeeds, lateral movement is a probable next step.',
    },
    {
      horizon: '+4',
      stage: 'Command & Control',
      probability: 48,
      confidence: 62,
      riskScore: 87,
      trafficPattern: 'Beaconing, periodic outbound connections',
      description: 'Model estimates possible establishment of C2 channel at this horizon.',
    },
    {
      horizon: '+5',
      stage: 'Exfiltration',
      probability: 34,
      confidence: 54,
      riskScore: 91,
      trafficPattern: 'Large outbound transfers, DNS tunneling signals',
      description: 'Distant forecast — lower confidence. Monitor for data movement indicators.',
    },
  ],
  10: [
    {
      horizon: 'Now',
      stage: 'Reconnaissance',
      probability: 72,
      confidence: 88,
      riskScore: 68,
      trafficPattern: 'High SYN ratio, port scan signatures',
      description: 'Observed behavior consistent with active network reconnaissance.',
    },
    {
      horizon: '+1',  stage: 'Scanning',        probability: 79, confidence: 86, riskScore: 73, trafficPattern: 'Service enumeration', description: 'Systematic scanning predicted.' },
    { horizon: '+2',  stage: 'Initial Access',   probability: 82, confidence: 81, riskScore: 78, trafficPattern: 'Auth failures', description: 'Access attempts estimated.' },
    { horizon: '+3',  stage: 'Lateral Movement', probability: 61, confidence: 71, riskScore: 83, trafficPattern: 'SMB pivoting',  description: 'Possible lateral spread.' },
    { horizon: '+4',  stage: 'Command & Control',probability: 48, confidence: 62, riskScore: 87, trafficPattern: 'Beaconing',     description: 'C2 channel possible.' },
    { horizon: '+5',  stage: 'Exfiltration',     probability: 34, confidence: 54, riskScore: 91, trafficPattern: 'Large outbound', description: 'Exfil risk rising.' },
    { horizon: '+6',  stage: 'Impact',           probability: 27, confidence: 44, riskScore: 93, trafficPattern: 'Disruption indicators', description: 'Low confidence impact forecast.' },
    { horizon: '+7',  stage: 'Impact',           probability: 22, confidence: 38, riskScore: 94, trafficPattern: 'Continued disruption', description: 'Extended impact estimate.' },
    { horizon: '+8',  stage: 'Impact',           probability: 18, confidence: 33, riskScore: 95, trafficPattern: 'Persistent disruption', description: 'Very distant — treat as indicative only.' },
    { horizon: '+9',  stage: 'Impact',           probability: 14, confidence: 28, riskScore: 95, trafficPattern: 'Unknown', description: 'Model uncertainty is high beyond this horizon.' },
    { horizon: '+10', stage: 'Impact',           probability: 11, confidence: 22, riskScore: 95, trafficPattern: 'Unknown', description: 'Beyond reliable forecast window.' },
  ],
};

export const mockTimelineEvents = [
  { id: 'e1', time: '09:32', label: 'Normal Traffic',             type: 'observed'  as const, stage: 'Benign',          detail: 'Baseline network activity. No anomalies detected.' },
  { id: 'e2', time: '09:35', label: 'Port Scan Detected',         type: 'observed'  as const, stage: 'Reconnaissance',  detail: 'Sequential port probing from 192.0.2.14. 847 ports scanned in 180s.' },
  { id: 'e3', time: '09:37', label: 'Suspicious SYN Activity',    type: 'observed'  as const, stage: 'Scanning',        detail: 'SYN ratio elevated to 0.74. Possible SYN sweep in progress.' },
  { id: 'e4', time: '09:39', label: 'Credential Activity',        type: 'observed'  as const, stage: 'Initial Access',  detail: 'Multiple authentication failures observed on host-beta.internal.' },
  { id: 'e5', time: '09:41', label: 'Initial Access Suspected',   type: 'observed'  as const, stage: 'Initial Access',  detail: 'Unusual authentication pattern. Model confidence: 78%.' },
  { id: 'e6', time: '09:43', label: 'Lateral Movement Forecast',  type: 'forecast'  as const, stage: 'Lateral Movement',detail: 'Model estimates 61% probability of lateral movement in the next window.' },
  { id: 'e7', time: '09:45', label: 'C2 Establishment Forecast',  type: 'forecast'  as const, stage: 'Command & Control',detail: 'If lateral movement occurs, C2 beaconing predicted with 48% probability.' },
];

export const mockMitreStages = [
  {
    id: 'recon',   tactic: 'Reconnaissance',     technique: 'Active Scanning',          status: 'observed'  as const, confidence: 88, evidence: 'Port sweep detected from 192.0.2.14. 847 ports in 3 min.', description: 'Adversary is actively probing the network to identify open services and hosts.' },
  { id: 'disc',    tactic: 'Discovery',           technique: 'Network Service Discovery',status: 'observed'  as const, confidence: 79, evidence: 'Service banner grabbing detected on TCP/22, TCP/443.', description: 'Active enumeration of network services to identify attack surface.' },
  { id: 'ia',      tactic: 'Initial Access',      technique: 'Credential-Based Access',  status: 'current'   as const, confidence: 82, evidence: 'Multiple auth failures on host-beta.internal. Brute-force pattern.', description: 'Attempts to use credentials to gain initial foothold on the network.' },
  { id: 'exec',    tactic: 'Execution',           technique: 'Command Execution',        status: 'forecast'  as const, confidence: 61, evidence: 'No direct evidence. Model extrapolation from current trajectory.', description: 'Predicted execution of adversary-controlled code following access.' },
  { id: 'persist', tactic: 'Persistence',         technique: 'Scheduled Task / Job',     status: 'not_observed' as const, confidence: 0, evidence: 'No indicators observed.', description: 'Technique not yet detected in current network session.' },
  { id: 'priv',    tactic: 'Privilege Escalation',technique: 'Exploitation for Privilege',status: 'not_observed' as const, confidence: 0, evidence: 'No indicators observed.', description: 'Technique not yet detected in current network session.' },
  { id: 'lat',     tactic: 'Lateral Movement',    technique: 'Internal Pivoting',        status: 'forecast'  as const, confidence: 55, evidence: 'No direct evidence. Predicted based on observed trajectory.', description: 'Model predicts movement from initial foothold to adjacent internal hosts.' },
  { id: 'c2',      tactic: 'Command and Control', technique: 'Encrypted Channel',        status: 'forecast'  as const, confidence: 44, evidence: 'No direct evidence. Distant forecast — treat as indicative.', description: 'Predicted establishment of persistent command channel to adversary infrastructure.' },
  { id: 'exfil',   tactic: 'Exfiltration',        technique: 'Data Transfer to External',status: 'not_observed' as const, confidence: 0, evidence: 'No indicators observed.', description: 'Technique not yet detected.' },
  { id: 'impact',  tactic: 'Impact',              technique: 'Network Denial of Service',status: 'not_observed' as const, confidence: 0, evidence: 'No indicators observed.', description: 'Technique not yet detected.' },
];
