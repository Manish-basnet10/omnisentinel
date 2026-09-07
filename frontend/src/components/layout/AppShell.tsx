import React from 'react';
import { Sidebar } from './Sidebar';
import { Header } from './Header';
import { ActiveDatasetBanner } from './ActiveDatasetBanner';

interface AppShellProps {
  children: React.ReactNode;
  alertCount?: number;
}

export function AppShell({ children, alertCount }: AppShellProps) {
  return (
    <div className="flex h-screen bg-surface-0 overflow-hidden">
      <Sidebar />
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
        <ActiveDatasetBanner />
        <Header alertCount={alertCount} />
        <main className="flex-1 overflow-y-auto p-4 lg:p-6 page-enter" role="main">
          {children}
        </main>
      </div>
    </div>
  );
}
