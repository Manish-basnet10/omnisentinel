import { mockFlows, mockLiveStats, mockProtocolDistribution, mockTopPorts, mockTcpFlags, generateFlows } from '../data/mockTraffic';
import client from './client';

const delay = (ms: number) => new Promise(r => setTimeout(r, ms));

// GET /network/live
export async function getLiveNetworkStats() {
  await delay(300);
  return {
    ...mockLiveStats,
    activeFlows:      mockLiveStats.activeFlows + Math.floor(Math.random() * 30 - 15),
    suspiciousFlows:  mockLiveStats.suspiciousFlows + Math.floor(Math.random() * 10 - 5),
    totalConnections: mockLiveStats.totalConnections + Math.floor(Math.random() * 50),
  };
}

// GET /network/traffic
export async function getNetworkFlows(page = 1, pageSize = 20) {
  await delay(400);
  const start = (page - 1) * pageSize;
  const flows = mockFlows.slice(start, start + pageSize);
  return { flows, total: mockFlows.length, page, pageSize };
}

// GET /network/protocols
export async function getProtocolDistribution() {
  await delay(250);
  return mockProtocolDistribution;
}

// GET /network/top-ports
export async function getTopPorts() {
  await delay(250);
  return mockTopPorts;
}

// GET /network/tcp-flags
export async function getTcpFlags() {
  await delay(200);
  return mockTcpFlags;
}

// uploadTrafficFile is now removed, handled by useAnalysis in the global provider
// Live update simulation
export function subscribeToLiveFlows(callback: (flows: ReturnType<typeof generateFlows>) => void) {
  const interval = setInterval(() => {
    callback(generateFlows(5)); // 5 new flows every tick
  }, 7000);
  return () => clearInterval(interval);
}
