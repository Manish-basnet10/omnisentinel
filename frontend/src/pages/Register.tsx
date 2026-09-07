import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { Shield, Zap, Eye, EyeOff } from 'lucide-react';
import { useAuth } from '../hooks/useAuth';

export default function Register() {
  const navigate = useNavigate();
  const [name, setName]         = useState('');
  const [email, setEmail]       = useState('');
  const [password, setPassword] = useState('');
  const [remember, setRemember] = useState(false);
  const [showPw, setShowPw]     = useState(false);
  const [loading, setLoading]   = useState(false);
  const [error, setError]       = useState('');

  const { register } = useAuth();

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!name || !email || !password) { setError('Please fill in all fields.'); return; }
    setError('');
    setLoading(true);
    
    try {
      await register(name, email, password);
      navigate('/dashboard');
    } catch (err: any) {
      setError(err.message || 'Registration failed.');
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen bg-surface-0 flex overflow-hidden">
      {/* Left Panel */}
      <div className="hidden lg:flex lg:w-3/5 flex-col justify-between p-14 bg-surface-1 border-r border-border-subtle relative overflow-hidden">
        {/* Background grid */}
        <div className="absolute inset-0 bg-grid-pattern bg-grid opacity-30 pointer-events-none" />

        {/* Brand */}
        <div className="relative">
          <div className="flex items-center gap-3 mb-8">
            <div className="w-10 h-10 rounded-lg bg-forecast-dim border border-forecast-dim flex items-center justify-center">
              <Shield size={20} className="text-forecast" />
            </div>
            <div>
              <div className="text-lg font-bold tracking-widest text-text-primary">PROJECT POSSIBLE</div>
              <div className="text-[11px] text-forecast tracking-widest uppercase">AI Network Defence</div>
            </div>
          </div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1, duration: 0.5 }}
          >
            <h1 className="text-4xl font-bold text-text-primary leading-tight mb-4">
              Don't Just Detect.<br />
              <span className="text-gradient-forecast">Anticipate.</span>
            </h1>
            <p className="text-lg text-text-secondary mb-2">AI-Powered Network Attack Forecasting</p>
            <p className="text-sm text-text-muted max-w-md leading-relaxed">
              Detect the present. Understand the sequence.<br />
              Anticipate what comes next.
            </p>
          </motion.div>
        </div>

        {/* Pipeline */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.4, duration: 0.6 }}
          className="relative"
        >
          <div className="text-[10px] uppercase tracking-widest text-text-disabled mb-4">Core Intelligence Pipeline</div>
          <div className="flex items-center gap-0">
            {['OBSERVE', 'UNDERSTAND', 'FORECAST', 'EXPLAIN', 'RESPOND'].map((step, i) => (
              <React.Fragment key={step}>
                <div className={`px-3 py-2 rounded text-[11px] font-semibold tracking-wide
                  ${step === 'FORECAST'
                    ? 'bg-forecast-dim border border-forecast text-forecast-bright'
                    : 'bg-surface-3 border border-border-subtle text-text-secondary'
                  }`}
                >
                  {step}
                </div>
                {i < 4 && <div className="w-6 h-px bg-border-emphasis mx-1" />}
              </React.Fragment>
            ))}
          </div>
          <p className="text-[11px] text-text-muted mt-3">
            Traditional tools stop at OBSERVE. Project Possible goes all the way to FORECAST.
          </p>
        </motion.div>

        {/* Stats row */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.6, duration: 0.5 }}
          className="grid grid-cols-3 gap-4 relative"
        >
          {[
            { value: '0.8153', label: 'Rollout AUC (k=1)', note: 'GRU World Model' },
            { value: '8 steps', label: 'Forecast Horizon', note: 'Autoregressive rollout' },
            { value: '43.6 pts', label: 'Risk Separation', note: 'Attack vs benign' },
          ].map(stat => (
            <div key={stat.label} className="border border-border-subtle rounded p-3 bg-surface-2">
              <div className="text-xl font-bold text-forecast font-mono">{stat.value}</div>
              <div className="text-[11px] text-text-secondary mt-0.5">{stat.label}</div>
              <div className="text-[10px] text-text-muted">{stat.note}</div>
            </div>
          ))}
        </motion.div>

        {/* Demo indicator */}
        <div className="relative text-[10px] text-text-disabled flex items-center gap-2">
          <Zap size={10} className="text-elevated" />
          DEMO MODE · Synthetic Data · Not Real Network Traffic
        </div>
      </div>

      {/* Right Panel — Login Form */}
      <div className="flex-1 flex items-center justify-center p-8">
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.4 }}
          className="w-full max-w-sm"
        >
          {/* Mobile brand */}
          <div className="lg:hidden mb-8">
            <div className="text-xl font-bold text-text-primary">PROJECT POSSIBLE</div>
            <div className="text-[11px] text-forecast">AI Network Attack Forecasting</div>
          </div>

          <h2 className="text-xl font-semibold text-text-primary mb-1">Create an Account</h2>
          <p className="text-sm text-text-muted mb-6">Join the SOC workspace</p>

          <form onSubmit={handleSubmit} className="space-y-4" noValidate>
            <div>
              <label htmlFor="name" className="block text-xs text-text-secondary mb-1.5">
                Full Name
              </label>
              <input
                id="name"
                type="text"
                autoComplete="name"
                value={name}
                onChange={e => setName(e.target.value)}
                placeholder="John Doe"
                className="input"
              />
            </div>

            <div>
              <label htmlFor="email" className="block text-xs text-text-secondary mb-1.5">
                Email address
              </label>
              <input
                id="email"
                type="email"
                autoComplete="email"
                value={email}
                onChange={e => setEmail(e.target.value)}
                placeholder="analyst@soc.example"
                className="input"
              />
            </div>

            <div>
              <label htmlFor="password" className="block text-xs text-text-secondary mb-1.5">
                Password
              </label>
              <div className="relative">
                <input
                  id="password"
                  type={showPw ? 'text' : 'password'}
                  autoComplete="current-password"
                  value={password}
                  onChange={e => setPassword(e.target.value)}
                  placeholder="••••••••"
                  className="input pr-10"
                />
                <button
                  type="button"
                  onClick={() => setShowPw(p => !p)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-text-disabled hover:text-text-secondary"
                  aria-label={showPw ? 'Hide password' : 'Show password'}
                >
                  {showPw ? <EyeOff size={14} /> : <Eye size={14} />}
                </button>
              </div>
            </div>

            <div className="flex items-center justify-between">
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={remember}
                  onChange={e => setRemember(e.target.checked)}
                  className="w-3.5 h-3.5 rounded border-border-default bg-surface-3 accent-forecast"
                />
                <span className="text-xs text-text-secondary">Remember me</span>
              </label>
              <button type="button" className="text-xs text-forecast hover:text-forecast-bright">
                Demo credentials accepted
              </button>
            </div>

            {error && (
              <div className="text-xs text-critical bg-critical-bg border border-critical-dim rounded px-3 py-2">
                {error}
              </div>
            )}

            <button
              type="submit"
              disabled={loading}
              className="btn btn-primary w-full justify-center h-10"
              id="signin-btn"
            >
              {loading ? (
                <span className="flex items-center gap-2">
                  <svg className="animate-spin w-4 h-4" viewBox="0 0 24 24" fill="none">
                    <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2" strokeOpacity="0.2" />
                    <path d="M12 2a10 10 0 0 1 10 10" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
                  </svg>
                  Authenticating…
                </span>
              ) : 'Sign Up'}
            </button>
          </form>

          <div className="mt-6 p-3 bg-surface-2 border border-border-subtle rounded text-[11px] text-text-muted">
            <span className="text-elevated font-semibold">Demo Mode:</span> Any credentials will sign you in. This is a synthetic data demonstration.
          </div>
          
          <div className="mt-4 text-center text-sm text-text-secondary">
            Already have an account? <button type="button" onClick={() => navigate('/login')} className="text-forecast hover:text-forecast-bright hover:underline">Sign in</button>
          </div>
        </motion.div>
      </div>
    </div>
  );
}
