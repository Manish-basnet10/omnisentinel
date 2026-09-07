/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      },
      colors: {
        // Surface layers
        surface: {
          0: '#0a0d12',
          1: '#0e1218',
          2: '#12171f',
          3: '#171d27',
          4: '#1c2330',
          5: '#222b3a',
        },
        // Borders
        border: {
          subtle: '#1e2a3a',
          default: '#243040',
          emphasis: '#2e3d52',
        },
        // Text
        text: {
          primary: '#e2e8f0',
          secondary: '#94a3b8',
          muted: '#64748b',
          disabled: '#3d4f65',
        },
        // Semantic
        critical: {
          DEFAULT: '#ef4444',
          dim: '#7f1d1d',
          bg: '#1a0505',
        },
        warning: {
          DEFAULT: '#f97316',
          dim: '#7c2d12',
          bg: '#1a0800',
        },
        elevated: {
          DEFAULT: '#eab308',
          dim: '#713f12',
          bg: '#1a1200',
        },
        safe: {
          DEFAULT: '#22c55e',
          dim: '#14532d',
          bg: '#051210',
        },
        info: {
          DEFAULT: '#38bdf8',
          dim: '#0c4a6e',
          bg: '#030f1a',
        },
        // AI / Forecast — restrained purple
        forecast: {
          DEFAULT: '#a78bfa',
          dim: '#4c1d95',
          bg: '#0d0520',
          bright: '#c4b5fd',
        },
      },
      backgroundImage: {
        'grid-pattern': "linear-gradient(rgba(30,42,58,0.4) 1px, transparent 1px), linear-gradient(90deg, rgba(30,42,58,0.4) 1px, transparent 1px)",
      },
      backgroundSize: {
        'grid': '32px 32px',
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'fade-in': 'fadeIn 0.3s ease-in-out',
        'slide-in': 'slideIn 0.3s ease-out',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0', transform: 'translateY(4px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        slideIn: {
          '0%': { opacity: '0', transform: 'translateX(-8px)' },
          '100%': { opacity: '1', transform: 'translateX(0)' },
        },
      },
    },
  },
  plugins: [],
};
