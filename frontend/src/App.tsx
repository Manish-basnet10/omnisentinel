import React from 'react';
import { BrowserRouter } from 'react-router-dom';
import { AuthProvider } from './hooks/useAuth';
import { AnalysisProvider } from './hooks/useAnalysis';
import { AppRoutes } from './routes/AppRoutes';

export default function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <AnalysisProvider>
          <AppRoutes />
        </AnalysisProvider>
      </AuthProvider>
    </BrowserRouter>
  );
}
