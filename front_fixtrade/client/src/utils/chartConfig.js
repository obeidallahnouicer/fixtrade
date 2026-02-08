// Professional chart configuration for trading platform
export const chartColors = {
  primary: '#3b82f6',
  success: '#10b981',
  danger: '#ef4444',
  warning: '#f59e0b',
  info: '#06b6d4',
  grid: '#1e293b',
  text: '#94a3b8',
  background: '#0f1117',
};

export const chartConfig = {
  style: {
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
  },
  cartesianGrid: {
    strokeDasharray: '3 3',
    stroke: chartColors.grid,
    opacity: 0.3,
  },
  axis: {
    stroke: chartColors.text,
    fontSize: 12,
    fontWeight: 400,
  },
  tooltip: {
    contentStyle: {
      backgroundColor: '#1a1d2a',
      border: '1px solid #2a2e3a',
      borderRadius: '8px',
      boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)',
    },
    labelStyle: {
      color: '#e5e7eb',
      fontWeight: 500,
    },
    itemStyle: {
      color: '#cbd5e1',
    },
  },
  legend: {
    wrapperStyle: {
      paddingTop: '10px',
    },
    iconType: 'line',
  },
};

// Chart line configurations
export const lineConfigs = {
  historical: {
    type: 'monotone',
    stroke: chartColors.primary,
    strokeWidth: 2,
    dot: false,
    activeDot: { r: 6, strokeWidth: 2 },
  },
  prediction: {
    type: 'monotone',
    stroke: chartColors.success,
    strokeWidth: 2,
    strokeDasharray: '5 5',
    dot: { fill: chartColors.success, r: 4 },
    activeDot: { r: 6, strokeWidth: 2 },
  },
  ma5: {
    type: 'monotone',
    stroke: chartColors.warning,
    strokeWidth: 1.5,
    dot: false,
  },
  ma10: {
    type: 'monotone',
    stroke: chartColors.info,
    strokeWidth: 1.5,
    dot: false,
  },
};

export default chartConfig;
