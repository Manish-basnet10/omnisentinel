// ─────────────────────────────────────────────────────────
// Mock Alerts Data  —  DEMO / SYNTHETIC DATA ONLY
// ─────────────────────────────────────────────────────────

export type AlertSeverity = 'Critical' | 'High' | 'Medium' | 'Low';
export type AlertStatus   = 'New' | 'Acknowledged' | 'Dismissed' | 'Investigating';

export interface Alert {
  id: string;
  severity: AlertSeverity;
  title: string;
  source: string;
  stage: string;
  confidence: number;
  time: string;
  status: AlertStatus;
  detail: string;
}

export const mockAlerts: Alert[] = [
  {
    id: 'ALT-0041',
    severity: 'Critical',
    title: 'High-confidence port scan from external host',
    source: '192.0.2.14',
    stage: 'Reconnaissance',
    confidence: 94,
    time: '09:35:12',
    status: 'New',
    detail: '847 TCP ports scanned in 180s. Matches SYN sweep pattern. Source not in allow-list.',
  },
  {
    id: 'ALT-0042',
    severity: 'Critical',
    title: 'Repeated authentication failures — possible brute-force',
    source: '192.0.2.87',
    stage: 'Initial Access',
    confidence: 88,
    time: '09:38:44',
    status: 'New',
    detail: '42 failed login attempts on host-beta.internal over SSH in 120s. Threshold exceeded.',
  },
  {
    id: 'ALT-0043',
    severity: 'High',
    title: 'Elevated SYN ratio — possible SYN sweep',
    source: '198.51.100.5',
    stage: 'Scanning',
    confidence: 81,
    time: '09:37:03',
    status: 'Investigating',
    detail: 'SYN/total ratio 0.74, significantly above baseline of 0.18. Duration: 3m 12s.',
  },
  {
    id: 'ALT-0044',
    severity: 'High',
    title: 'Unusual destination host diversity',
    source: '192.0.2.142',
    stage: 'Reconnaissance',
    confidence: 76,
    time: '09:33:21',
    status: 'Acknowledged',
    detail: 'Single source contacted 94 distinct internal hosts in 10 minutes. Anomalous for this subnet.',
  },
  {
    id: 'ALT-0045',
    severity: 'High',
    title: 'Forecast: Lateral movement probability elevated',
    source: 'AI Forecast Engine',
    stage: 'Lateral Movement',
    confidence: 61,
    time: '09:40:00',
    status: 'New',
    detail: 'Model estimates 61% probability of lateral movement in the next observation window. Treat as predictive signal.',
  },
  {
    id: 'ALT-0046',
    severity: 'Medium',
    title: 'Suspicious outbound DNS query volume',
    source: '10.0.0.12',
    stage: 'Reconnaissance',
    confidence: 64,
    time: '09:29:55',
    status: 'Acknowledged',
    detail: 'Internal host querying 312 unique external domains in 15 minutes. DNS beaconing possible.',
  },
  {
    id: 'ALT-0047',
    severity: 'Medium',
    title: 'FTP session to non-standard port',
    source: '198.51.100.44',
    stage: 'Initial Access',
    confidence: 58,
    time: '09:27:14',
    status: 'Dismissed',
    detail: 'FTP connection established to destination on port 2121. Non-standard port warrants review.',
  },
  {
    id: 'ALT-0048',
    severity: 'Low',
    title: 'Inbound traffic from low-reputation address space',
    source: '203.0.113.11',
    stage: 'Benign',
    confidence: 41,
    time: '09:21:33',
    status: 'Dismissed',
    detail: 'Connection from address with low historical reputation. No attack indicators currently observed.',
  },
  {
    id: 'ALT-0049',
    severity: 'Low',
    title: 'Inter-arrival time deviation from baseline',
    source: '192.0.2.201',
    stage: 'Scanning',
    confidence: 38,
    time: '09:19:07',
    status: 'Acknowledged',
    detail: 'IAT standard deviation 4.2x higher than rolling 24h baseline. Weak scanning signal.',
  },
  {
    id: 'ALT-0050',
    severity: 'Medium',
    title: 'Multiple protocols from single source',
    source: '198.51.100.178',
    stage: 'Discovery',
    confidence: 55,
    time: '09:15:49',
    status: 'New',
    detail: 'Source used TCP, UDP, ICMP, and DNS within a 5-minute window. Possible mixed-protocol enumeration.',
  },
];

export const mockInvestigation = {
  id: 'PP-1042',
  title: 'Sustained Reconnaissance and Credential Activity',
  status: 'Active',
  riskScore: 78,
  riskLevel: 'HIGH',
  createdAt: '2026-09-06 09:32:00',
  updatedAt: '2026-09-06 09:41:00',
  analyst: 'Security Analyst',
  affectedAssets: ['host-beta.internal', 'host-alpha.internal', '10.0.0.5'],
  observedBehavior: [
    'Active port scanning from 192.0.2.14 — 847 ports in 180 seconds',
    'SYN ratio elevated to 0.74 (baseline: 0.18)',
    '42 failed SSH authentications on host-beta.internal',
    'Destination host diversity: 94 unique internal hosts from single source',
  ],
  predictedBehavior: [
    'Probability of transition to Initial Access: 82% (Model Estimate)',
    'If access is obtained, Lateral Movement predicted with 61% probability',
    'C2 channel establishment estimated at 48% within 5 observation windows',
  ],
  evidence: [
    { id: 'ev1', type: 'Network Flow', description: 'TCP SYN sweep — 192.0.2.14 → 10.0.0.0/24', time: '09:35' },
    { id: 'ev2', type: 'Auth Log',    description: 'SSH auth failures — host-beta.internal',      time: '09:38' },
    { id: 'ev3', type: 'Alert',       description: 'ALT-0041: High-confidence port scan',          time: '09:35' },
    { id: 'ev4', type: 'Alert',       description: 'ALT-0042: Brute-force pattern detected',       time: '09:38' },
  ],
  recommendedSteps: [
    'Isolate 192.0.2.14 traffic at perimeter firewall pending review',
    'Review SSH access logs on host-beta.internal for the past 60 minutes',
    'Enable enhanced packet capture on the 10.0.0.0/24 subnet',
    'Alert senior analyst if probability exceeds 90% on next forecast window',
    'Document all observed indicators for threat intelligence update',
  ],
  aiExplanation: {
    prediction: 'Initial Access',
    confidence: 82,
    factors: [
      { feature: 'SYN Packet Ratio',  contribution: 31, direction: 'positive' as const },
      { feature: 'Port Diversity',     contribution: 24, direction: 'positive' as const },
      { feature: 'Destination Count',  contribution: 18, direction: 'positive' as const },
      { feature: 'Inter-arrival Time', contribution: 13, direction: 'negative' as const },
      { feature: 'Connection Rate',    contribution: 9,  direction: 'positive' as const },
    ],
  },
};
