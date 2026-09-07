import { mockDashboardSummary, mockForecastHero, mockAttackTrajectory, mockRiskTimeSeries, mockExplainability, mockTrafficOverview } from '../data/mockDashboard';

const delay = (ms: number) => new Promise(r => setTimeout(r, ms));

// GET /dashboard/summary
export async function getDashboardSummary() {
  await delay(400);
  // Add tiny random drift so values feel live
  return {
    ...mockDashboardSummary,
    networkRisk:    mockDashboardSummary.networkRisk + Math.floor(Math.random() * 4 - 2),
    suspiciousFlows: mockDashboardSummary.suspiciousFlows + Math.floor(Math.random() * 20 - 10),
    lastUpdated: new Date().toISOString(),
  };
}

export async function getForecastHero() {
  await delay(300);
  return { ...mockForecastHero, timestamp: new Date().toISOString() };
}

export async function getAttackTrajectory() {
  await delay(200);
  return mockAttackTrajectory;
}

export async function getRiskTimeSeries() {
  await delay(300);
  return mockRiskTimeSeries;
}

export async function getExplainability() {
  await delay(250);
  return mockExplainability;
}

export async function getTrafficOverview(window: '1H' | '6H' | '24H' | '7D') {
  await delay(300);
  return mockTrafficOverview[window];
}
