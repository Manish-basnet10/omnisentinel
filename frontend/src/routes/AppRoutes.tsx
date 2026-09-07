import React, { Suspense } from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { useAuth } from '../hooks/useAuth';
import { Spinner } from '../components/common/States';

// Lazy loaded pages
const Login = React.lazy(() => import('../pages/Login'));
const Register = React.lazy(() => import('../pages/Register'));
const Dashboard = React.lazy(() => import('../pages/Dashboard'));
const LiveNetwork = React.lazy(() => import('../pages/LiveNetwork'));
const TrafficAnalysis = React.lazy(() => import('../pages/TrafficAnalysis'));
const Alerts = React.lazy(() => import('../pages/Alerts'));
const AttackForecast = React.lazy(() => import('../pages/AttackForecast'));
const AttackTimeline = React.lazy(() => import('../pages/AttackTimeline'));
const MitreAttack = React.lazy(() => import('../pages/MitreAttack'));
const ModelPerformance = React.lazy(() => import('../pages/ModelPerformance'));
const Explainability = React.lazy(() => import('../pages/Explainability'));
const Investigations = React.lazy(() => import('../pages/Investigations'));
const Reports = React.lazy(() => import('../pages/Reports'));
const Settings = React.lazy(() => import('../pages/Settings'));

// Protected route wrapper
function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const { isAuthenticated } = useAuth();
  if (!isAuthenticated) return <Navigate to="/login" replace />;
  return <>{children}</>;
}

function LoadingFallback() {
  return (
    <div className="flex items-center justify-center h-screen w-full bg-surface-0">
      <div className="flex flex-col items-center gap-4">
        <Spinner size={32} />
        <div className="text-text-muted text-xs tracking-widest uppercase animate-pulse">Loading Interface</div>
      </div>
    </div>
  );
}

export function AppRoutes() {
  return (
    <Suspense fallback={<LoadingFallback />}>
      <Routes>
        <Route path="/login" element={<Login />} />
        <Route path="/register" element={<Register />} />
        
        {/* Protected Routes */}
        <Route path="/" element={<ProtectedRoute><Navigate to="/dashboard" replace /></ProtectedRoute>} />
        <Route path="/dashboard" element={<ProtectedRoute><Dashboard /></ProtectedRoute>} />
        <Route path="/live-network" element={<ProtectedRoute><LiveNetwork /></ProtectedRoute>} />
        <Route path="/traffic-analysis" element={<ProtectedRoute><TrafficAnalysis /></ProtectedRoute>} />
        <Route path="/alerts" element={<ProtectedRoute><Alerts /></ProtectedRoute>} />
        <Route path="/attack-forecast" element={<ProtectedRoute><AttackForecast /></ProtectedRoute>} />
        <Route path="/attack-timeline" element={<ProtectedRoute><AttackTimeline /></ProtectedRoute>} />
        <Route path="/mitre-attack" element={<ProtectedRoute><MitreAttack /></ProtectedRoute>} />
        <Route path="/model-performance" element={<ProtectedRoute><ModelPerformance /></ProtectedRoute>} />
        <Route path="/explainability" element={<ProtectedRoute><Explainability /></ProtectedRoute>} />
        <Route path="/investigations" element={<ProtectedRoute><Investigations /></ProtectedRoute>} />
        <Route path="/reports" element={<ProtectedRoute><Reports /></ProtectedRoute>} />
        <Route path="/settings" element={<ProtectedRoute><Settings /></ProtectedRoute>} />
        
        {/* Catch all */}
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Suspense>
  );
}
