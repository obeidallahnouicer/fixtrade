import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';

dotenv.config();

const app = express();
const PORT = process.env.PORT || 5000;
const BVMT_API_URL = process.env.BVMT_API_URL || 'https://www.bvmt.com.tn/rest_api/rest/market/groups/11,12,52,95,99';
const FIXTRADE_API_URL = process.env.FIXTRADE_API_URL || 'http://localhost:8000';

// Middleware
app.use(cors());
app.use(express.json());

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

// Proxy endpoint for BVMT stocks API
app.get('/api/stocks', async (req, res) => {
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 10000); // 10 second timeout

    const response = await fetch(BVMT_API_URL, {
      signal: controller.signal,
      headers: {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
      },
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      throw new Error(`BVMT API responded with status ${response.status}`);
    }

    const data = await response.json();
    res.json(data);
  } catch (error) {
    console.error('Error fetching from BVMT API:', error.message);
    
    if (error.name === 'AbortError') {
      return res.status(504).json({ 
        error: 'Request timeout', 
        message: 'The BVMT API took too long to respond' 
      });
    }
    
    if (error.message.includes('fetch failed')) {
      return res.status(503).json({ 
        error: 'Network error', 
        message: 'Unable to connect to BVMT API' 
      });
    }
    
    res.status(500).json({ 
      error: 'Internal server error', 
      message: error.message 
    });
  }
});

// ── Generic Fixtrade proxy helper ────────────────────────────────────
async function proxyToFixtrade(req, res, { path, method = 'GET', timeout = 15000 }) {
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeout);

    const opts = {
      method,
      signal: controller.signal,
      headers: { 'Content-Type': 'application/json' },
    };
    if (method !== 'GET' && method !== 'HEAD') {
      opts.body = JSON.stringify(req.body);
    }

    // Build target URL, preserving query string
    const qs = new URL(req.url, 'http://localhost').search;
    const targetUrl = `${FIXTRADE_API_URL}${path}${qs}`;

    const response = await fetch(targetUrl, opts);
    clearTimeout(timeoutId);

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      return res.status(response.status).json(errorData);
    }

    const data = await response.json();
    res.json(data);
  } catch (error) {
    console.error(`Proxy error [${method} ${path}]:`, error.message);
    if (error.name === 'AbortError') {
      return res.status(504).json({ error: 'Request timeout', message: 'Backend took too long to respond' });
    }
    if (error.message.includes('fetch failed') || error.code === 'ECONNREFUSED') {
      return res.status(503).json({ error: 'Service unavailable', message: 'Unable to connect to Fixtrade API. Make sure the backend is running on port 8000.' });
    }
    res.status(500).json({ error: 'Internal server error', message: error.message });
  }
}

// ── Trading API proxies ──────────────────────────────────────────────

// Price predictions
app.post('/api/v1/trading/predictions', (req, res) =>
  proxyToFixtrade(req, res, { path: '/api/v1/trading/predictions', method: 'POST' })
);

// Sentiment
app.post('/api/v1/trading/sentiment', (req, res) =>
  proxyToFixtrade(req, res, { path: '/api/v1/trading/sentiment', method: 'POST' })
);

// Anomalies (detect)
app.post('/api/v1/trading/anomalies', (req, res) =>
  proxyToFixtrade(req, res, { path: '/api/v1/trading/anomalies', method: 'POST' })
);

// Anomalies (recent – GET)
app.get('/api/v1/trading/anomalies/recent', (req, res) =>
  proxyToFixtrade(req, res, { path: '/api/v1/trading/anomalies/recent', method: 'GET' })
);

// Trade recommendation
app.post('/api/v1/trading/recommendations', (req, res) =>
  proxyToFixtrade(req, res, { path: '/api/v1/trading/recommendations', method: 'POST' })
);

// Volume prediction
app.post('/api/v1/trading/predictions/volume', (req, res) =>
  proxyToFixtrade(req, res, { path: '/api/v1/trading/predictions/volume', method: 'POST' })
);

// Liquidity prediction
app.post('/api/v1/trading/predictions/liquidity', (req, res) =>
  proxyToFixtrade(req, res, { path: '/api/v1/trading/predictions/liquidity', method: 'POST' })
);

// Sentiment article analysis
app.post('/api/v1/trading/sentiment/analyze', (req, res) =>
  proxyToFixtrade(req, res, { path: '/api/v1/trading/sentiment/analyze', method: 'POST' })
);

// ── AI Agent API proxies ─────────────────────────────────────────────

// AI status
app.get('/api/v1/ai/status', (req, res) =>
  proxyToFixtrade(req, res, { path: '/api/v1/ai/status', method: 'GET' })
);

// Profile questionnaire
app.post('/api/v1/ai/profile/questionnaire', (req, res) =>
  proxyToFixtrade(req, res, { path: '/api/v1/ai/profile/questionnaire', method: 'POST' })
);

// Portfolio CRUD
app.post('/api/v1/ai/portfolio/create', (req, res) =>
  proxyToFixtrade(req, res, { path: '/api/v1/ai/portfolio/create', method: 'POST' })
);

app.get('/api/v1/ai/portfolio/:id/snapshot', (req, res) =>
  proxyToFixtrade(req, res, { path: `/api/v1/ai/portfolio/${req.params.id}/snapshot`, method: 'GET' })
);

app.get('/api/v1/ai/portfolio/:id/performance', (req, res) =>
  proxyToFixtrade(req, res, { path: `/api/v1/ai/portfolio/${req.params.id}/performance`, method: 'GET' })
);

app.get('/api/v1/ai/portfolio/:id/performance/explain', (req, res) =>
  proxyToFixtrade(req, res, { path: `/api/v1/ai/portfolio/${req.params.id}/performance/explain`, method: 'GET' })
);

app.get('/api/v1/ai/portfolio/:id/position/:symbol', (req, res) =>
  proxyToFixtrade(req, res, { path: `/api/v1/ai/portfolio/${req.params.id}/position/${req.params.symbol}`, method: 'GET' })
);

// Trades
app.post('/api/v1/ai/portfolio/:id/trade', (req, res) =>
  proxyToFixtrade(req, res, { path: `/api/v1/ai/portfolio/${req.params.id}/trade`, method: 'POST' })
);

// Price updates
app.post('/api/v1/ai/portfolio/:id/prices/update', (req, res) =>
  proxyToFixtrade(req, res, { path: `/api/v1/ai/portfolio/${req.params.id}/prices/update`, method: 'POST' })
);

// Stop-loss check
app.post('/api/v1/ai/portfolio/:id/stop-loss/check', (req, res) =>
  proxyToFixtrade(req, res, { path: `/api/v1/ai/portfolio/${req.params.id}/stop-loss/check`, method: 'POST' })
);

// AI recommendations
app.get('/api/v1/ai/recommendations', (req, res) =>
  proxyToFixtrade(req, res, { path: '/api/v1/ai/recommendations', method: 'GET' })
);

app.get('/api/v1/ai/recommendations/:symbol/explain', (req, res) =>
  proxyToFixtrade(req, res, { path: `/api/v1/ai/recommendations/${req.params.symbol}/explain`, method: 'GET' })
);

app.listen(PORT, () => {
  console.log(`🚀 Proxy server running on http://localhost:${PORT}`);
  console.log(`📊 BVMT API: ${BVMT_API_URL}`);
  console.log(`🤖 Fixtrade API: ${FIXTRADE_API_URL}`);
});
