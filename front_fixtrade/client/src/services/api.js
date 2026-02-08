import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:5000/api';
const API_TIMEOUT = 10000; // 10 seconds

// Create axios instance
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: API_TIMEOUT,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor
apiClient.interceptors.request.use(
  (config) => {
    // You can add auth tokens here if needed
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor
apiClient.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    // Classify error types
    if (error.code === 'ECONNABORTED' || error.code === 'ERR_NETWORK') {
      console.error('Network error:', error.message);
      return Promise.reject({
        type: 'network',
        message: 'Unable to connect to server. Please check your connection.',
        originalError: error,
      });
    }

    if (error.response) {
      // Server responded with error status
      const status = error.response.status;
      
      if (status === 504) {
        return Promise.reject({
          type: 'timeout',
          message: 'Request timed out. Please try again.',
          originalError: error,
        });
      }

      if (status === 503) {
        return Promise.reject({
          type: 'service',
          message: 'Service temporarily unavailable.',
          originalError: error,
        });
      }

      return Promise.reject({
        type: 'api',
        message: error.response.data?.message || 'An error occurred',
        status,
        originalError: error,
      });
    }

    return Promise.reject({
      type: 'unknown',
      message: 'An unexpected error occurred',
      originalError: error,
    });
  }
);

/**
 * Fetch stock data from BVMT API via proxy
 * @returns {Promise<Object>} Stock data response
 */
export const fetchStocks = async () => {
  try {
    const response = await apiClient.get('/stocks');
    return response.data;
  } catch (error) {
    console.error('Error fetching stocks:', error);
    throw error;
  }
};

/**
 * Health check endpoint
 * @returns {Promise<Object>} Health status
 */
export const checkHealth = async () => {
  try {
    const response = await apiClient.get('/health');
    return response.data;
  } catch (error) {
    console.error('Health check failed:', error);
    throw error;
  }
};

// ── Trading API ──────────────────────────────────────────────────────

/** Predict stock price for a given symbol and horizon */
export const predictStockPrice = async (symbol, horizonDays = 1) => {
  const response = await apiClient.post('/v1/trading/predictions', {
    symbol,
    horizon_days: horizonDays,
  });
  return response.data;
};

/** Predict transaction volume */
export const predictVolume = async (symbol, horizonDays = 5) => {
  const response = await apiClient.post('/v1/trading/predictions/volume', {
    symbol,
    horizon_days: horizonDays,
  });
  return response.data;
};

/** Predict liquidity tiers */
export const predictLiquidity = async (symbol, horizonDays = 5) => {
  const response = await apiClient.post('/v1/trading/predictions/liquidity', {
    symbol,
    horizon_days: horizonDays,
  });
  return response.data;
};

/** Get aggregated sentiment for a stock symbol */
export const getStockSentiment = async (symbol, targetDate = null) => {
  const response = await apiClient.post('/v1/trading/sentiment', {
    symbol,
    ...(targetDate && { target_date: targetDate }),
  });
  return response.data;
};

/** Detect anomalies for a specific stock */
export const detectAnomalies = async (symbol) => {
  const response = await apiClient.post('/v1/trading/anomalies', { symbol });
  return response.data;
};

/** Get recent anomaly alerts across all stocks */
export const getRecentAnomalies = async ({ symbol = null, limit = 50, hoursBack = 24 } = {}) => {
  const response = await apiClient.get('/v1/trading/anomalies/recent', {
    params: {
      ...(symbol && { symbol }),
      limit,
      hours_back: hoursBack,
    },
  });
  return response.data;
};

/** Get trade recommendation for a stock */
export const getTradeRecommendation = async (symbol, portfolioId = 'default') => {
  const response = await apiClient.post('/v1/trading/recommendations', {
    symbol,
    portfolio_id: portfolioId,
  });
  return response.data;
};

// ── AI Agent API ─────────────────────────────────────────────────────

/** Get AI module status */
export const getAIStatus = async () => {
  const response = await apiClient.get('/v1/ai/status');
  return response.data;
};

/** Submit risk profile questionnaire */
export const submitProfileQuestionnaire = async (answers) => {
  const response = await apiClient.post('/v1/ai/profile/questionnaire', answers);
  return response.data;
};

/** Create a new portfolio */
export const createPortfolio = async (riskProfile = 'moderate', initialCapital = 10000) => {
  const response = await apiClient.post('/v1/ai/portfolio/create', {
    risk_profile: riskProfile,
    initial_capital: initialCapital,
  });
  return response.data;
};

/** Get portfolio snapshot */
export const getPortfolioSnapshot = async (portfolioId = 'default') => {
  const response = await apiClient.get(`/v1/ai/portfolio/${portfolioId}/snapshot`);
  return response.data;
};

/** Get portfolio performance metrics */
export const getPortfolioPerformance = async (portfolioId = 'default') => {
  const response = await apiClient.get(`/v1/ai/portfolio/${portfolioId}/performance`);
  return response.data;
};

/** Get AI explanation of portfolio performance */
export const getPerformanceExplanation = async (portfolioId = 'default') => {
  const response = await apiClient.get(`/v1/ai/portfolio/${portfolioId}/performance/explain`);
  return response.data;
};

/** Get position for a symbol in a portfolio */
export const getPosition = async (portfolioId, symbol) => {
  const response = await apiClient.get(`/v1/ai/portfolio/${portfolioId}/position/${symbol}`);
  return response.data;
};

/** Execute a trade */
export const executeTrade = async (portfolioId, { symbol, action, quantity, price, generateExplanation = true }) => {
  const response = await apiClient.post(`/v1/ai/portfolio/${portfolioId}/trade`, {
    symbol,
    action,
    quantity,
    price,
    generate_explanation: generateExplanation,
  });
  return response.data;
};

/** Update market prices for portfolio */
export const updatePortfolioPrices = async (portfolioId, prices) => {
  const response = await apiClient.post(`/v1/ai/portfolio/${portfolioId}/prices/update`, prices);
  return response.data;
};

/** Check stop-losses for portfolio */
export const checkStopLosses = async (portfolioId, prices) => {
  const response = await apiClient.post(`/v1/ai/portfolio/${portfolioId}/stop-loss/check`, prices);
  return response.data;
};

/** Get AI daily recommendations */
export const getAIRecommendations = async ({ portfolioId = 'default', topN = 10, symbols = null } = {}) => {
  const response = await apiClient.get('/v1/ai/recommendations', {
    params: {
      portfolio_id: portfolioId,
      top_n: topN,
      ...(symbols && { symbols: symbols.join(',') }),
    },
  });
  return response.data;
};

/** Get detailed AI explanation for a stock */
export const getRecommendationExplanation = async (symbol, portfolioId = 'default') => {
  const response = await apiClient.get(`/v1/ai/recommendations/${symbol}/explain`, {
    params: { portfolio_id: portfolioId },
  });
  return response.data;
};

export default apiClient;
