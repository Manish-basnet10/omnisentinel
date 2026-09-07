export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  return `${(bytes / 1024 / 1024 / 1024).toFixed(2)} GB`;
}

export function formatNumber(n: number): string {
  return n.toLocaleString('en-US');
}

export function formatPct(n: number, digits = 0): string {
  return `${n.toFixed(digits)}%`;
}

export function formatDuration(seconds: number): string {
  if (seconds < 60) return `${seconds}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
  return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`;
}

export function getTimeLabel(): string {
  return new Date().toLocaleTimeString('en-GB', { hour12: false });
}

export function getDateTimeLabel(): string {
  return new Date().toLocaleString('en-GB', {
    weekday: 'long',
    year: 'numeric',
    month: 'long',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false,
  });
}

export function riskLevelFromScore(score: number): 'CRITICAL' | 'HIGH' | 'MEDIUM' | 'LOW' | 'SAFE' {
  if (score >= 80) return 'CRITICAL';
  if (score >= 65) return 'HIGH';
  if (score >= 45) return 'MEDIUM';
  if (score >= 25) return 'LOW';
  return 'SAFE';
}

export function riskColorClass(score: number): string {
  if (score >= 80) return 'text-critical';
  if (score >= 65) return 'text-warning';
  if (score >= 45) return 'text-elevated';
  if (score >= 25) return 'text-info';
  return 'text-safe';
}

export function confidenceLabel(pct: number): string {
  if (pct >= 85) return 'Very High';
  if (pct >= 70) return 'High';
  if (pct >= 55) return 'Moderate';
  if (pct >= 40) return 'Low';
  return 'Very Low';
}
