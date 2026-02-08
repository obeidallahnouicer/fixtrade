import { useState, useMemo, useEffect } from 'react';
import { useStockData } from '../hooks/useStockData';
import { useRecentAnomalies } from '../hooks/useAnomalies';
import { BarChart3, TrendingUp, Newspaper, Bell, Search, Activity, AlertTriangle, RefreshCw } from 'lucide-react';

const Alerts = () => {
  const { stocks, isLoading: isStocksLoading } = useStockData();
  const [hoursBack, setHoursBack] = useState(24);
  const { anomalies, isLoading: isAnomaliesLoading, isError, refetch } = useRecentAnomalies({ limit: 100, hoursBack });

  const [selectedTypes, setSelectedTypes] = useState(['price', 'volume', 'anomaly']);
  const [selectedSeverity, setSelectedSeverity] = useState(['low', 'medium', 'high', 'critical']);
  const [searchQuery, setSearchQuery] = useState('');

  // Current time tick (used for relative timestamps)
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), 60000);
    return () => clearInterval(id);
  }, []);

  // Combine backend anomalies with live stock alerts (large variation stocks)
  const allAlerts = useMemo(() => {
    const alerts = [];

    // Add real anomalies from backend
    if (anomalies && anomalies.length > 0) {
      anomalies.forEach((a, idx) => {
        alerts.push({
          id: `anomaly_${a.id || idx}`,
          type: 'anomaly',
          severity: a.severity || 'medium',
          ticker: a.symbol || '–',
          name: stocks?.find(s => s.ticker === a.symbol)?.societe || a.symbol,
          title: a.anomaly_type || 'Anomalie Détectée',
          message: a.description || 'Anomalie détectée par le modèle IA.',
          timestamp: a.detected_at ? new Date(a.detected_at).getTime() : now,
          value: null,
          source: 'backend',
        });
      });
    }

    // Add live price alerts from current stock data
    if (stocks && stocks.length > 0) {
      stocks.forEach((stock) => {
        const variation = parseFloat(stock.variation) || 0;
        if (Math.abs(variation) > 3) {
          alerts.push({
            id: `live_price_${stock.ticker}`,
            type: 'price',
            severity: Math.abs(variation) > 5 ? 'high' : 'medium',
            ticker: stock.ticker,
            name: stock.societe || stock.stockName,
            title: variation > 0 ? 'Forte Hausse de Prix' : 'Forte Baisse de Prix',
            message: `${stock.ticker} a ${variation > 0 ? 'gagné' : 'perdu'} ${Math.abs(variation).toFixed(2)}% aujourd'hui`,
            timestamp: now,
            value: `${variation >= 0 ? '+' : ''}${variation.toFixed(2)}%`,
            source: 'live',
          });
        }

        // Volume alerts – stocks with unusually high volume
        const vol = parseFloat(stock.volumeD) || 0;
        if (vol > 100000) {
          alerts.push({
            id: `live_vol_${stock.ticker}`,
            type: 'volume',
            severity: vol > 500000 ? 'high' : 'medium',
            ticker: stock.ticker,
            name: stock.societe || stock.stockName,
            title: 'Volume Élevé',
            message: `${stock.ticker} affiche un volume de ${vol.toLocaleString('fr-FR')} titres`,
            timestamp: now,
            value: vol.toLocaleString('fr-FR'),
            source: 'live',
          });
        }
      });
    }

    // Sort by severity then timestamp
    const sevOrder = { critical: 0, high: 1, medium: 2, low: 3 };
    return alerts.sort((a, b) => (sevOrder[a.severity] ?? 4) - (sevOrder[b.severity] ?? 4) || b.timestamp - a.timestamp);
  }, [anomalies, stocks, now]);

  // Filter alerts
  const filteredAlerts = useMemo(() => {
    return allAlerts.filter(alert => {
      const typeMatch = selectedTypes.includes(alert.type);
      const sevMatch = selectedSeverity.includes(alert.severity);
      const searchMatch = searchQuery === '' ||
        alert.ticker.toLowerCase().includes(searchQuery.toLowerCase()) ||
        alert.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        alert.message.toLowerCase().includes(searchQuery.toLowerCase());
      return typeMatch && sevMatch && searchMatch;
    });
  }, [allAlerts, selectedTypes, selectedSeverity, searchQuery]);

  const toggleFilter = (filterType, value) => {
    if (filterType === 'type') {
      setSelectedTypes(prev =>
        prev.includes(value) ? prev.filter(t => t !== value) : [...prev, value]
      );
    } else if (filterType === 'severity') {
      setSelectedSeverity(prev =>
        prev.includes(value) ? prev.filter(s => s !== value) : [...prev, value]
      );
    }
  };

  const getSeverityColor = (severity) => {
    switch (severity) {
      case 'critical': return 'border-danger-500 bg-danger-500/10';
      case 'high': return 'border-warning-500 bg-warning-500/10';
      case 'medium': return 'border-primary-500 bg-primary-500/10';
      case 'low':
      default: return 'border-finance-border bg-finance-bg';
    }
  };

  const getTypeIcon = (type) => {
    switch (type) {
      case 'volume': return BarChart3;
      case 'price': return TrendingUp;
      case 'anomaly': return AlertTriangle;
      default: return Bell;
    }
  };

  const formatTimestamp = (ts) => {
    const diff = now - ts;
    const minutes = Math.floor(diff / 60000);
    const hours = Math.floor(diff / 3600000);
    if (minutes < 1) return 'À l\'instant';
    if (minutes < 60) return `Il y a ${minutes} min`;
    if (hours < 24) return `Il y a ${hours}h`;
    return new Date(ts).toLocaleDateString('fr-FR', { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' });
  };

  // Stats
  const alertStats = useMemo(() => ({
    total: allAlerts.length,
    anomaly: allAlerts.filter(a => a.type === 'anomaly').length,
    price: allAlerts.filter(a => a.type === 'price').length,
    volume: allAlerts.filter(a => a.type === 'volume').length,
    critical: allAlerts.filter(a => a.severity === 'critical' || a.severity === 'high').length,
  }), [allAlerts]);

  const isLoading = isStocksLoading || isAnomaliesLoading;

  if (isLoading) {
    return (
      <div className="p-4 md:p-6 lg:p-8">
        <div className="max-w-[1600px] mx-auto">
          <div className="animate-pulse space-y-6">
            <div className="h-12 bg-finance-bg rounded w-1/3"></div>
            <div className="grid grid-cols-5 gap-4">
              {[1, 2, 3, 4, 5].map(i => (
                <div key={i} className="h-24 bg-finance-bg rounded"></div>
              ))}
            </div>
            <div className="space-y-4">
              {[1, 2, 3].map(i => (
                <div key={i} className="h-32 bg-finance-bg rounded"></div>
              ))}
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="p-4 md:p-6 lg:p-8">
      <div className="max-w-[1600px] mx-auto">
        {/* Header */}
        <div className="mb-6 flex items-center justify-between">
          <div>
            <h1 className="text-2xl md:text-3xl font-bold text-finance-text-primary mb-2">
              Surveillance & Alertes
            </h1>
            <p className="text-finance-text-secondary">
              Détection d'anomalies en temps réel — Données live + IA
            </p>
          </div>
          <button
            onClick={() => refetch()}
            className="px-4 py-2 bg-finance-card border border-finance-border rounded-lg text-finance-text-primary hover:bg-finance-bg transition-colors flex items-center gap-2"
          >
            <RefreshCw className="w-4 h-4" />
            Actualiser
          </button>
        </div>

        {/* Alert Statistics */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4 mb-6">
          <div className="finance-card p-4 rounded-lg">
            <p className="text-sm text-finance-text-secondary mb-1">Total Alertes</p>
            <p className="text-2xl font-bold text-finance-text-primary">{alertStats.total}</p>
          </div>
          <div className="finance-card p-4 rounded-lg">
            <p className="text-sm text-finance-text-secondary mb-1">Critique / Haute</p>
            <p className="text-2xl font-bold text-danger-500">{alertStats.critical}</p>
          </div>
          <div className="finance-card p-4 rounded-lg">
            <p className="text-sm text-finance-text-secondary mb-1 flex items-center gap-1">
              <AlertTriangle className="w-4 h-4" /> Anomalies IA
            </p>
            <p className="text-2xl font-bold text-finance-text-primary">{alertStats.anomaly}</p>
          </div>
          <div className="finance-card p-4 rounded-lg">
            <p className="text-sm text-finance-text-secondary mb-1 flex items-center gap-1">
              <TrendingUp className="w-4 h-4" /> Prix
            </p>
            <p className="text-2xl font-bold text-finance-text-primary">{alertStats.price}</p>
          </div>
          <div className="finance-card p-4 rounded-lg">
            <p className="text-sm text-finance-text-secondary mb-1 flex items-center gap-1">
              <BarChart3 className="w-4 h-4" /> Volume
            </p>
            <p className="text-2xl font-bold text-finance-text-primary">{alertStats.volume}</p>
          </div>
        </div>

        {/* Filters & Search */}
        <div className="finance-card p-6 rounded-lg mb-6">
          <h3 className="text-lg font-semibold text-finance-text-primary mb-4">Filtres</h3>

          <div className="mb-4 relative">
            <Search className="absolute left-3 top-3.5 w-5 h-5 text-finance-text-secondary" />
            <input
              type="text"
              placeholder="Rechercher par ticker, nom ou message..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full px-4 py-3 pl-10 bg-finance-bg border border-finance-border rounded-lg text-finance-text-primary focus:outline-none focus:ring-2 focus:ring-primary-500"
            />
          </div>

          <div className="mb-4">
            <p className="text-sm text-finance-text-secondary mb-2">Type d'Alerte</p>
            <div className="flex flex-wrap gap-2">
              {[
                { value: 'anomaly', label: 'Anomalies IA', count: alertStats.anomaly },
                { value: 'price', label: 'Prix', count: alertStats.price },
                { value: 'volume', label: 'Volume', count: alertStats.volume },
              ].map(filter => (
                <button
                  key={filter.value}
                  onClick={() => toggleFilter('type', filter.value)}
                  className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                    selectedTypes.includes(filter.value)
                      ? 'bg-primary-500 text-white'
                      : 'bg-finance-bg text-finance-text-secondary hover:bg-finance-border'
                  }`}
                >
                  {filter.label} ({filter.count})
                </button>
              ))}
            </div>
          </div>

          <div className="flex items-center gap-4">
            <div>
              <p className="text-sm text-finance-text-secondary mb-2">Sévérité</p>
              <div className="flex flex-wrap gap-2">
                {[
                  { value: 'critical', label: 'Critique' },
                  { value: 'high', label: 'Haute' },
                  { value: 'medium', label: 'Moyenne' },
                  { value: 'low', label: 'Basse' },
                ].map(filter => (
                  <button
                    key={filter.value}
                    onClick={() => toggleFilter('severity', filter.value)}
                    className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                      selectedSeverity.includes(filter.value)
                        ? 'bg-primary-500 text-white'
                        : 'bg-finance-bg text-finance-text-secondary hover:bg-finance-border'
                    }`}
                  >
                    {filter.label}
                  </button>
                ))}
              </div>
            </div>
            <div className="ml-auto">
              <p className="text-sm text-finance-text-secondary mb-2">Période</p>
              <select
                value={hoursBack}
                onChange={e => setHoursBack(Number(e.target.value))}
                className="px-3 py-2 bg-finance-bg border border-finance-border rounded text-finance-text-primary"
              >
                <option value={1}>Dernière heure</option>
                <option value={6}>6 heures</option>
                <option value={24}>24 heures</option>
                <option value={72}>3 jours</option>
                <option value={168}>7 jours</option>
              </select>
            </div>
          </div>
        </div>

        {/* Error state */}
        {isError && (
          <div className="finance-card p-4 rounded-lg mb-6 border border-warning-500 bg-warning-500/10">
            <p className="text-finance-text-primary">
              Impossible de charger les anomalies du backend. Les alertes live sont toujours disponibles.
            </p>
          </div>
        )}

        {/* Real-time Feed */}
        <div className="finance-card p-6 rounded-lg">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-finance-text-primary flex items-center gap-2">
              <Activity className="w-5 h-5 text-primary-500" />
              Feed en Temps Réel
            </h3>
            <span className="text-sm text-finance-text-secondary">
              {filteredAlerts.length} alerte(s) affichée(s)
            </span>
          </div>

          {filteredAlerts.length === 0 ? (
            <div className="text-center py-12">
              <Search className="w-16 h-16 mx-auto mb-4 text-finance-text-secondary" />
              <p className="text-finance-text-secondary">Aucune alerte ne correspond à vos filtres</p>
            </div>
          ) : (
            <div className="space-y-3 max-h-[600px] overflow-y-auto">
              {filteredAlerts.map((alert) => (
                <div
                  key={alert.id}
                  className={`p-4 rounded-lg border ${getSeverityColor(alert.severity)} transition-all hover:shadow-lg`}
                >
                  <div className="flex items-start justify-between gap-4">
                    <div className="flex items-start gap-3 flex-1">
                      {(() => {
                        const IconComponent = getTypeIcon(alert.type);
                        return <IconComponent className="w-5 h-5 mt-0.5" />;
                      })()}
                      <div className="flex-1">
                        <div className="flex items-center gap-2 mb-1">
                          <span className="font-bold text-finance-text-primary">{alert.ticker}</span>
                          <span className="text-sm text-finance-text-secondary">{alert.name}</span>
                          {alert.source === 'backend' && (
                            <span className="text-xs px-2 py-0.5 rounded bg-primary-500/20 text-primary-400">IA</span>
                          )}
                        </div>
                        <p className="font-semibold text-finance-text-primary mb-1">{alert.title}</p>
                        <p className="text-sm text-finance-text-secondary mb-2">{alert.message}</p>
                        <div className="flex items-center gap-4 text-xs text-finance-text-secondary">
                          <span>{formatTimestamp(alert.timestamp)}</span>
                          {alert.value && (
                            <>
                              <span>·</span>
                              <span className="font-semibold">{alert.value}</span>
                            </>
                          )}
                          <span>·</span>
                          <span className="capitalize">{alert.severity}</span>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Alerts;
