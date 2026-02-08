import { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { BarChart3, TrendingUp, Briefcase, Bell, Brain, Settings, HelpCircle } from 'lucide-react';

const Sidebar = () => {
  const [isCollapsed, setIsCollapsed] = useState(false);
  const location = useLocation();

  const menuItems = [
    { id: 'market', label: 'Vue d\'Ensemble', icon: BarChart3, path: '/market' },
    { id: 'analysis', label: 'Analyse Valeur', icon: TrendingUp, path: '/analysis' },
    { id: 'portfolio', label: 'Mon Portefeuille', icon: Briefcase, path: '/portfolio' },
    { id: 'alerts', label: 'Alertes', icon: Bell, path: '/alerts' },
    { id: 'ai', label: 'Agent IA', icon: Brain, path: '/ai' },
  ];

  const bottomItems = [
    { id: 'settings', label: 'Paramètres', icon: Settings },
    { id: 'help', label: 'Aide', icon: HelpCircle },
  ];

  return (
    <div 
      className={`finance-card rounded-none border-r border-finance-border h-screen sticky top-0 flex flex-col transition-all duration-300 ${
        isCollapsed ? 'w-16' : 'w-64'
      }`}
    >
      {/* Header */}
      <div className="p-4 border-b border-finance-border flex items-center justify-between">
        {!isCollapsed && (
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 bg-primary-500 rounded flex items-center justify-center text-white font-bold">
              FT
            </div>
            <div>
              <h1 className="text-sm font-semibold text-finance-text-primary">FixTrade</h1>
              <p className="text-xs text-finance-text-secondary">BVMT Dashboard</p>
            </div>
          </div>
        )}
        <button
          onClick={() => setIsCollapsed(!isCollapsed)}
          className="p-1.5 hover:bg-finance-bg rounded transition-colors"
        >
          <svg 
            className={`w-5 h-5 text-finance-text-secondary transition-transform ${isCollapsed ? 'rotate-180' : ''}`}
            fill="none" 
            stroke="currentColor" 
            viewBox="0 0 24 24"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 19l-7-7 7-7m8 14l-7-7 7-7" />
          </svg>
        </button>
      </div>

      {/* Navigation */}
      <nav className="flex-1 p-3 overflow-y-auto">
        <div className="space-y-1">
          {menuItems.map((item) => {
            const IconComponent = item.icon;
            const isActive = location.pathname === item.path;
            return (
              <Link
                key={item.id}
                to={item.path}
                className={`w-full flex items-center gap-3 px-3 py-2.5 rounded transition-colors ${
                  isActive
                    ? 'bg-primary-500/10 text-primary-400 border border-primary-500/20'
                    : 'text-finance-text-secondary hover:bg-finance-bg hover:text-finance-text-primary'
                }`}
                title={isCollapsed ? item.label : ''}
              >
                <IconComponent className="w-5 h-5" />
                {!isCollapsed && (
                  <span className="text-sm font-medium">{item.label}</span>
                )}
              </Link>
            );
          })}
        </div>
      </nav>

      {/* Bottom Section */}
      <div className="p-3 border-t border-finance-border">
        <div className="space-y-1 mb-3">
          {bottomItems.map((item) => {
            const IconComponent = item.icon;
            return (
              <button
                key={item.id}
                onClick={() => {}}
                className="w-full flex items-center gap-3 px-3 py-2.5 rounded text-finance-text-secondary hover:bg-finance-bg hover:text-finance-text-primary transition-colors"
                title={isCollapsed ? item.label : ''}
              >
                <IconComponent className="w-5 h-5" />
                {!isCollapsed && (
                  <span className="text-sm font-medium">{item.label}</span>
                )}
              </button>
            );
          })}
        </div>

        {/* User Profile */}
        {!isCollapsed && (
          <div className="p-3 bg-finance-bg rounded flex items-center gap-3">
            <div className="w-8 h-8 bg-primary-500 rounded-full flex items-center justify-center text-white text-sm font-semibold">
              U
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium text-finance-text-primary truncate">User</p>
              <p className="text-xs text-finance-text-secondary truncate">trader@bvmt.tn</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default Sidebar;
