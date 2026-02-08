import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import MarketOverview from './pages/MarketOverview';
import StockAnalysis from './pages/StockAnalysis';
import Portfolio from './pages/Portfolio';
import Alerts from './pages/Alerts';
import AIProfile from './pages/AIProfile';
import './App.css';

// Create a client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: true,
      refetchOnReconnect: true,
      retry: 3,
      staleTime: 25000,
    },
  },
});

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <Router>
        <div className="App flex min-h-screen bg-[#0f1117]">
          <Sidebar />
          <div className="flex-1 overflow-auto">
            <Routes>
              <Route path="/" element={<Navigate to="/market" replace />} />
              <Route path="/market" element={<MarketOverview />} />
              <Route path="/analysis" element={<StockAnalysis />} />
              <Route path="/portfolio" element={<Portfolio />} />
              <Route path="/alerts" element={<Alerts />} />
              <Route path="/ai" element={<AIProfile />} />
              <Route path="*" element={<Navigate to="/market" replace />} />
            </Routes>
          </div>
        </div>
      </Router>
    </QueryClientProvider>
  );
}

export default App
