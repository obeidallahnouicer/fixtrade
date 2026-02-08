import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  createPortfolio,
  getPortfolioSnapshot,
  getPortfolioPerformance,
  getPerformanceExplanation,
  executeTrade,
  updatePortfolioPrices,
  checkStopLosses,
} from '../services/api';

const PORTFOLIO_ID_KEY = 'fixtrade_portfolio_id';

/** Get stored portfolio id from localStorage */
export const getStoredPortfolioId = () =>
  localStorage.getItem(PORTFOLIO_ID_KEY) || null;

/** Store portfolio id */
export const setStoredPortfolioId = (id) =>
  localStorage.setItem(PORTFOLIO_ID_KEY, id);

/**
 * Hook to manage portfolio creation and state persistence.
 */
export const useCreatePortfolio = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ riskProfile, initialCapital }) =>
      createPortfolio(riskProfile, initialCapital),
    onSuccess: (data) => {
      if (data?.portfolio_id) {
        setStoredPortfolioId(data.portfolio_id);
        queryClient.invalidateQueries({ queryKey: ['portfolio'] });
      }
    },
  });
};

/**
 * Hook to fetch the current portfolio snapshot (positions, cash, value).
 */
export const usePortfolioSnapshot = (portfolioId) => {
  const pid = portfolioId || getStoredPortfolioId() || 'default';

  const query = useQuery({
    queryKey: ['portfolio', 'snapshot', pid],
    queryFn: () => getPortfolioSnapshot(pid),
    staleTime: 30000,
    retry: 1,
  });

  return {
    snapshot: query.data,
    isLoading: query.isLoading,
    isError: query.isError,
    error: query.error,
    refetch: query.refetch,
  };
};

/**
 * Hook to fetch portfolio performance metrics (ROI, Sharpe, drawdown…).
 */
export const usePortfolioPerformance = (portfolioId) => {
  const pid = portfolioId || getStoredPortfolioId() || 'default';

  const query = useQuery({
    queryKey: ['portfolio', 'performance', pid],
    queryFn: () => getPortfolioPerformance(pid),
    staleTime: 60000,
    retry: 1,
  });

  return {
    performance: query.data,
    isLoading: query.isLoading,
    isError: query.isError,
    refetch: query.refetch,
  };
};

/**
 * Hook to get AI-generated explanation of portfolio performance.
 */
export const usePerformanceExplanation = (portfolioId) => {
  const pid = portfolioId || getStoredPortfolioId() || 'default';

  const query = useQuery({
    queryKey: ['portfolio', 'explanation', pid],
    queryFn: () => getPerformanceExplanation(pid),
    staleTime: 300000, // 5 min – AI explanation is expensive
    retry: 1,
    enabled: !!pid,
  });

  return {
    explanation: query.data?.explanation || null,
    isLoading: query.isLoading,
    isError: query.isError,
  };
};

/**
 * Hook to execute trades (buy/sell).
 */
export const useExecuteTrade = (portfolioId) => {
  const queryClient = useQueryClient();
  const pid = portfolioId || getStoredPortfolioId() || 'default';

  return useMutation({
    mutationFn: (tradeData) => executeTrade(pid, tradeData),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['portfolio'] });
    },
  });
};

/**
 * Hook to update live prices and check stop-losses in one go.
 */
export const useUpdatePrices = (portfolioId) => {
  const queryClient = useQueryClient();
  const pid = portfolioId || getStoredPortfolioId() || 'default';

  return useMutation({
    mutationFn: async (prices) => {
      await updatePortfolioPrices(pid, prices);
      const stopLossResult = await checkStopLosses(pid, prices);
      return stopLossResult;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['portfolio'] });
    },
  });
};
