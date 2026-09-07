import { mockForecastSteps, mockTimelineEvents, mockMitreStages } from '../data/mockForecast';
import client from './client';

const delay = (ms: number) => new Promise(r => setTimeout(r, ms));

// POST /forecast — run the forecast pipeline (simulated)
export async function runForecast(horizon: number = 5) {
  // Simulate the multi-step inference pipeline
  const steps = [
    { step: 'Collecting network state',    duration: 600 },
    { step: 'Processing temporal sequence',duration: 800 },
    { step: 'Running model',               duration: 1200 },
    { step: 'Generating forecast',         duration: 600 },
    { step: 'Calculating confidence',      duration: 400 },
    { step: 'Generating explanation',      duration: 500 },
  ];
  return { steps, totalDuration: steps.reduce((a, s) => a + s.duration, 0) };
}

// GET /forecast/latest
export async function getLatestForecast(horizon: 1 | 3 | 5 | 10 = 5) {
  try {
    const res = await client.get('/api/forecasts');
    if (res.data && res.data.length > 0) {
      const latest = res.data[0];
      return {
        steps: latest.forecast_steps,
        horizon: latest.forecast_horizon,
        overallRisk: latest.current_risk,
        runAt: latest.timestamp,
        prediction: latest.predicted_stage,
      };
    }
  } catch (err) {
    console.error('Failed to get latest forecast, using mock', err);
  }

  await delay(350);
  const key = horizon as keyof typeof mockForecastSteps;
  return {
    steps: mockForecastSteps[key] || mockForecastSteps[5],
    horizon,
    overallRisk: 78,
    runAt: new Date().toISOString(),
  };
}

// GET /forecast/timeline
export async function getForecastTimeline() {
  await delay(300);
  return mockTimelineEvents;
}

// GET /mitre/stages
export async function getMitreStages() {
  await delay(400);
  return mockMitreStages;
}
