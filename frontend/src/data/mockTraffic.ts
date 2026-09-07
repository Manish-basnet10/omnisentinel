// ─────────────────────────────────────────────────────────
// Mock Traffic Data  —  DEMO / SYNTHETIC DATA ONLY
// Uses TEST-NET ranges (192.0.2.x, 198.51.100.x, 203.0.113.x)
// ─────────────────────────────────────────────────────────

export type RiskLevel = 'Critical' | 'High' | 'Medium' | 'Low' | 'Safe';
export type FlowStatus = 'Active' | 'Blocked' | 'Suspicious' | 'Normal';
export type Protocol = 'TCP' | 'UDP' | 'ICMP' | 'HTTP' | 'HTTPS' | 'DNS' | 'SSH' | 'FTP';

export interface NetworkFlow {
  id: string;
  timestamp: string;
  source: string;
  destination: string;
  protocol: Protocol;
  port: number;
  packets: number;
  bytes: number;
  risk: RiskLevel;
  status: FlowStatus;
  duration: number; // seconds
  flags?: string;
}

const SRC_HOSTS = [
  '192.0.2.14',  '192.0.2.87',  '192.0.2.142', '192.0.2.201',
  '198.51.100.5','198.51.100.44','198.51.100.99','198.51.100.178',
  '203.0.113.11','203.0.113.67',
];
const DST_HOSTS = [
  'host-alpha.internal',   'host-beta.internal',
  'host-gamma.internal',   'host-delta.internal',
  'host-epsilon.internal', 'host-zeta.internal',
  '10.0.0.1','10.0.0.5','10.0.0.12','10.0.0.22',
];
const PROTOCOLS: Protocol[] = ['TCP','UDP','ICMP','HTTP','HTTPS','DNS','SSH','FTP'];
const PORTS = [80,443,22,21,53,8080,8443,3389,445,139,25,587,993,110];
const RISK_LEVELS: RiskLevel[] = ['Safe','Low','Medium','High','Critical'];
const STATUSES: FlowStatus[] = ['Normal','Active','Suspicious','Blocked'];

function rng(min: number, max: number) {
  return Math.floor(Math.random() * (max - min + 1)) + min;
}

function pick<T>(arr: T[]): T {
  return arr[Math.floor(Math.random() * arr.length)];
}

function makeTimestamp(offsetMin: number) {
  const d = new Date();
  d.setMinutes(d.getMinutes() - offsetMin);
  return d.toLocaleTimeString('en-GB', { hour12: false });
}

export function generateFlows(count = 80): NetworkFlow[] {
  return Array.from({ length: count }, (_, i) => {
    const riskLevel = pick(RISK_LEVELS);
    const status: FlowStatus =
      riskLevel === 'Critical' ? 'Blocked' :
      riskLevel === 'High'     ? 'Suspicious' :
      riskLevel === 'Medium'   ? (Math.random() > 0.5 ? 'Suspicious' : 'Active') :
      'Normal';
    return {
      id: `flow-${Date.now()}-${i}`,
      timestamp: makeTimestamp(rng(0, 60)),
      source: pick(SRC_HOSTS),
      destination: pick(DST_HOSTS),
      protocol: pick(PROTOCOLS),
      port: pick(PORTS),
      packets: rng(1, 5000),
      bytes: rng(64, 1_500_000),
      risk: riskLevel,
      status,
      duration: rng(1, 3600),
      flags: pick(['SYN','SYN-ACK','ACK','FIN','RST','PSH-ACK','']),
    };
  });
}

export const mockFlows = generateFlows(80);

export const mockLiveStats = {
  totalConnections: 14_872,
  activeFlows: 1_284,
  suspiciousFlows: 247,
  blockedConnections: 38,
};

export const mockProtocolDistribution = [
  { name: 'TCP',   value: 48 },
  { name: 'HTTPS', value: 22 },
  { name: 'HTTP',  value: 12 },
  { name: 'UDP',   value: 9  },
  { name: 'DNS',   value: 5  },
  { name: 'SSH',   value: 3  },
  { name: 'Other', value: 1  },
];

export const mockTopPorts = {
  source: [
    { port: 49152, count: 3241 },
    { port: 49153, count: 2187 },
    { port: 49154, count: 1943 },
    { port: 49155, count: 1642 },
    { port: 49156, count: 1388 },
  ],
  destination: [
    { port: 443,  count: 6421 },
    { port: 80,   count: 3184 },
    { port: 22,   count: 1822 },
    { port: 53,   count: 1245 },
    { port: 8080, count: 934  },
  ],
};

export const mockTcpFlags = [
  { flag: 'SYN',     count: 4210 },
  { flag: 'ACK',     count: 8940 },
  { flag: 'SYN-ACK', count: 3847 },
  { flag: 'FIN',     count: 2140 },
  { flag: 'RST',     count: 1082 },
  { flag: 'PSH',     count: 3620 },
];
