import React, { createContext, useContext, useState, useCallback, useEffect } from 'react';
import client from '../api/client';

interface User { id: string; name: string; email: string; }
interface AuthContextValue {
  user: User | null;
  isAuthenticated: boolean;
  login: (email: string, password: string) => Promise<void>;
  register: (name: string, email: string, password: string) => Promise<void>;
  logout: () => void;
}

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(() => {
    const stored = localStorage.getItem('omnisentinel_user');
    return stored ? JSON.parse(stored) : null;
  });

  const checkAuth = useCallback(async () => {
    try {
      const res = await client.get('/auth/me');
      setUser(res.data);
      localStorage.setItem('omnisentinel_user', JSON.stringify(res.data));
    } catch (err) {
      setUser(null);
      localStorage.removeItem('omnisentinel_user');
      localStorage.removeItem('omnisentinel_token');
    }
  }, []);

  useEffect(() => {
    const token = localStorage.getItem('omnisentinel_token');
    if (token) {
      checkAuth();
    }
  }, [checkAuth]);

  const login = useCallback(async (email: string, password: string) => {
    const formData = new URLSearchParams();
    formData.append('username', email);
    formData.append('password', password);
    
    const res = await client.post('/auth/login', formData, {
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' }
    });
    
    localStorage.setItem('omnisentinel_token', res.data.access_token);
    await checkAuth();
  }, [checkAuth]);

  const register = useCallback(async (name: string, email: string, password: string) => {
    await client.post('/auth/register', { name, email, password });
    await login(email, password);
  }, [login]);

  const logout = useCallback(() => {
    setUser(null);
    localStorage.removeItem('omnisentinel_user');
    localStorage.removeItem('omnisentinel_token');
  }, []);

  return (
    <AuthContext.Provider value={{ user, isAuthenticated: !!user, login, register, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('useAuth must be used within AuthProvider');
  return ctx;
}
