import { mockAlerts, mockInvestigation, type Alert, type AlertStatus } from '../data/mockAlerts';
import client from './client';

const delay = (ms: number) => new Promise(r => setTimeout(r, ms));

let _alerts = [...mockAlerts];

// GET /alerts
export async function getAlerts(): Promise<Alert[]> {
  try {
    const res = await client.get('/api/alerts');
    return res.data.map((a: any) => ({
      id: a._id,
      time: new Date(a.timestamp).toLocaleTimeString(),
      severity: a.severity === 5 ? 'Critical' : a.severity === 4 ? 'High' : 'Medium',
      title: `Predicted ${a.mitre_tactic || 'Attack'} from PCAP`,
      source: 'PCAP Analysis',
      stage: a.current_stage || 'Unknown',
      confidence: Math.round((a.probability || 0.8) * 100),
      status: a.status === 'new' ? 'New' : a.status === 'investigating' ? 'Investigating' : a.status === 'resolved' ? 'Dismissed' : 'New',
      detail: `Risk Score: ${Math.round(a.risk_score || 0)}. Predicted stage: ${a.predicted_stage || 'Unknown'}`,
    }));
  } catch (err) {
    console.error('Failed to load alerts, using mock', err);
    return [..._alerts];
  }
}

// GET /alerts/:id
export async function getAlert(id: string) {
  await delay(200);
  return _alerts.find(a => a.id === id) ?? null;
}

// PATCH /alerts/:id/status
export async function updateAlertStatus(id: string, status: AlertStatus): Promise<Alert> {
  try {
    const backendStatus = status === 'New' ? 'new' : status === 'Investigating' ? 'investigating' : 'resolved';
    await client.patch(`/api/alerts/${id}`, { status: backendStatus });
  } catch (err) {
    console.error('Failed to update alert in backend', err);
  }
  
  _alerts = _alerts.map(a => a.id === id ? { ...a, status } : a);
  const updated = _alerts.find(a => a.id === id);
  if (!updated) throw new Error(`Alert ${id} not found`);
  return updated;
}

// GET /investigations
export async function getInvestigations() {
  await delay(300);
  return [mockInvestigation];
}

// GET /investigations/:id
export async function getInvestigation(id: string) {
  await delay(300);
  if (id === mockInvestigation.id || id === 'PP-1042') return mockInvestigation;
  return null;
}

// POST /investigations — create new
export async function createInvestigation(alertId: string) {
  await delay(500);
  return { ...mockInvestigation, id: `PP-${Math.floor(Math.random() * 9000 + 1000)}`, relatedAlert: alertId };
}

// Reset for demo purposes
export function resetAlerts() {
  _alerts = [...mockAlerts];
}
