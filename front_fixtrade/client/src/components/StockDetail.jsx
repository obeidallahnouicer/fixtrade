import { motion, AnimatePresence } from 'framer-motion';
import { formatCurrency, formatLargeNumber, formatTime } from '../utils/formatters';

const StockDetail = ({ stock, isOpen, onClose }) => {
  if (!stock) return null;

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="fixed inset-0 bg-black/50 backdrop-blur-sm z-40"
          />

          {/* Modal */}
          <motion.div
            initial={{ opacity: 0, scale: 0.9, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.9, y: 20 }}
            transition={{ type: 'spring', damping: 25, stiffness: 300 }}
            className="fixed inset-4 md:inset-auto md:left-1/2 md:top-1/2 md:-translate-x-1/2 md:-translate-y-1/2 md:w-full md:max-w-2xl z-50"
          >
            <div className="glass-card rounded-3xl p-6 md:p-8 h-full md:h-auto overflow-y-auto max-h-[90vh]">
              {/* Header */}
              <div className="flex items-start justify-between mb-6">
                <div>
                  <h2 className="text-2xl md:text-3xl font-bold text-gray-900 dark:text-white mb-1">
                    {stock.stockName}
                  </h2>
                  <p className="text-gray-600 dark:text-gray-400">
                    {stock.ticker} · ISIN: {stock.isin || 'N/A'}
                  </p>
                </div>
                <button
                  onClick={onClose}
                  className="p-2 rounded-full hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
                >
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>

              {/* Price Information */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                <div className="bg-white/10 dark:bg-black/20 rounded-xl p-4">
                  <div className="text-xs text-gray-600 dark:text-gray-400 mb-1">Last Price</div>
                  <div className="text-xl font-bold text-gray-900 dark:text-white">
                    {formatCurrency(stock.last)}
                  </div>
                </div>
                <div className="bg-white/10 dark:bg-black/20 rounded-xl p-4">
                  <div className="text-xs text-gray-600 dark:text-gray-400 mb-1">Open</div>
                  <div className="text-xl font-bold text-gray-900 dark:text-white">
                    {formatCurrency(stock.open)}
                  </div>
                </div>
                <div className="bg-white/10 dark:bg-black/20 rounded-xl p-4">
                  <div className="text-xs text-gray-600 dark:text-gray-400 mb-1">High</div>
                  <div className="text-xl font-bold text-appleGreen">
                    {formatCurrency(stock.high)}
                  </div>
                </div>
                <div className="bg-white/10 dark:bg-black/20 rounded-xl p-4">
                  <div className="text-xs text-gray-600 dark:text-gray-400 mb-1">Low</div>
                  <div className="text-xl font-bold text-appleRed">
                    {formatCurrency(stock.low)}
                  </div>
                </div>
              </div>

              {/* Volume and Market Data */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
                <div className="bg-white/10 dark:bg-black/20 rounded-xl p-4">
                  <div className="text-sm text-gray-600 dark:text-gray-400 mb-2">Trading Volume</div>
                  <div className="text-2xl font-bold text-gray-900 dark:text-white">
                    {formatLargeNumber(stock.volume)}
                  </div>
                  <div className="text-xs text-gray-500 dark:text-gray-500 mt-1">
                    Transactions: {stock.trVolume || 0}
                  </div>
                </div>
                <div className="bg-white/10 dark:bg-black/20 rounded-xl p-4">
                  <div className="text-sm text-gray-600 dark:text-gray-400 mb-2">Market Cap</div>
                  <div className="text-2xl font-bold text-gray-900 dark:text-white">
                    {formatLargeNumber(stock.caps)} TND
                  </div>
                  <div className="text-xs text-gray-500 dark:text-gray-500 mt-1">
                    Previous Close: {formatCurrency(stock.close)}
                  </div>
                </div>
              </div>

              {/* Order Book */}
              <div className="mb-6">
                <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-3">Order Book</h3>
                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-appleGreen/10 rounded-xl p-4">
                    <div className="text-sm font-medium text-appleGreen mb-2">Best Bid</div>
                    <div className="text-xl font-bold text-gray-900 dark:text-white">
                      {formatCurrency(stock.limit?.bid)}
                    </div>
                    <div className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                      Qty: {stock.limit?.bidQty || 0} · Orders: {stock.limit?.bidOrd || 0}
                    </div>
                  </div>
                  <div className="bg-appleRed/10 rounded-xl p-4">
                    <div className="text-sm font-medium text-appleRed mb-2">Best Ask</div>
                    <div className="text-xl font-bold text-gray-900 dark:text-white">
                      {formatCurrency(stock.limit?.ask)}
                    </div>
                    <div className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                      Qty: {stock.limit?.askQty || 0} · Orders: {stock.limit?.askOrd || 0}
                    </div>
                  </div>
                </div>
              </div>

              {/* Trading Info */}
              <div className="bg-white/10 dark:bg-black/20 rounded-xl p-4">
                <div className="grid grid-cols-2 gap-3 text-sm">
                  <div>
                    <span className="text-gray-600 dark:text-gray-400">Status:</span>
                    <span className="ml-2 font-semibold text-gray-900 dark:text-white">
                      {stock.status || 'Trading'}
                    </span>
                  </div>
                  <div>
                    <span className="text-gray-600 dark:text-gray-400">Session:</span>
                    <span className="ml-2 font-semibold text-gray-900 dark:text-white">
                      {stock.seance || 'N/A'}
                    </span>
                  </div>
                  <div>
                    <span className="text-gray-600 dark:text-gray-400">Last Update:</span>
                    <span className="ml-2 font-semibold text-gray-900 dark:text-white">
                      {formatTime(stock.time)}
                    </span>
                  </div>
                  <div>
                    <span className="text-gray-600 dark:text-gray-400">Group:</span>
                    <span className="ml-2 font-semibold text-gray-900 dark:text-white">
                      {stock.valGroup}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
};

export default StockDetail;
