import { useEffect, useState, useRef } from 'react';
import './App.css';

interface LogMessage {
  id: number;
  timestamp: string;
  raw_text: string;
  level: string;
  component: string;
}

interface AlertMessage {
  id: number;
  timestamp: string;
  risk_score: number;
  alert_level: string;
  triggering_logs?: string;
}

interface Stats {
  total_logs: number;
  active_alerts: number;
  latest_alert_level: string;
}

function App() {
  const [logs, setLogs] = useState<LogMessage[]>([]);
  const [alerts, setAlerts] = useState<AlertMessage[]>([]);
  const [stats, setStats] = useState<Stats>({ total_logs: 0, active_alerts: 0, latest_alert_level: 'None' });
  const [currentScore, setCurrentScore] = useState<number>(0);
  const [scoreHistory, setScoreHistory] = useState<{score: number, level: string}[]>([]);
  
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Initial fetch
    fetch('http://127.0.0.1:8000/stats')
      .then(r => r.json())
      .then(data => setStats(data))
      .catch(console.error);
      
    fetch('http://127.0.0.1:8000/recent_alerts?limit=10')
      .then(r => r.json())
      .then(data => setAlerts(data))
      .catch(console.error);

    fetch('http://127.0.0.1:8000/recent_logs?limit=50')
      .then(r => r.json())
      .then(data => setLogs(data.reverse()))
      .catch(console.error);

    // WebSocket connection
    const ws = new WebSocket('ws://127.0.0.1:8000/ws');
    
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === 'update') {
        const { log, risk_score, alert } = data;
        
        setLogs(prev => [...prev.slice(-100), log]);
        setCurrentScore(risk_score);
        
        let level = 'Low';
        if (risk_score > 80) level = 'Critical';
        else if (risk_score > 60) level = 'High';
        else if (risk_score > 30) level = 'Medium';
        
        setScoreHistory(prev => [...prev.slice(-30), { score: risk_score, level }]);
        
        setStats(prev => ({
          ...prev,
          total_logs: prev.total_logs + 1,
        }));

        if (alert) {
          setAlerts(prev => [alert, ...prev.slice(0, 49)]);
          setStats(prev => ({
            ...prev,
            active_alerts: prev.active_alerts + 1,
            latest_alert_level: alert.alert_level
          }));
        }
      }
    };

    return () => ws.close();
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  return (
    <div className="app-container">
      <header className="header">
        <h1>Neuro-Fuzzy SIEM Dashboard</h1>
        <div style={{ display: 'flex', alignItems: 'center', color: '#94a3b8' }}>
          <span className="live-indicator"></span>
          Streaming Active
        </div>
      </header>

      <main className="dashboard-content">
        {/* Left Column */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem', gridRow: '1 / span 2' }}>
          
          <div className="stats-grid">
            <div className="card">
              <h2>Total Events</h2>
              <div className="stat-value">{stats.total_logs.toLocaleString()}</div>
            </div>
            <div className="card">
              <h2>Active Alerts</h2>
              <div className="stat-value" style={{ color: 'var(--color-critical)' }}>
                {stats.active_alerts.toLocaleString()}
              </div>
            </div>
            <div className="card">
              <h2>Current Risk Score</h2>
              <div className="stat-value" style={{ 
                color: currentScore > 80 ? 'var(--color-critical)' : 
                       currentScore > 60 ? 'var(--color-high)' : 
                       currentScore > 30 ? 'var(--color-medium)' : 'var(--color-normal)'
              }}>
                {currentScore.toFixed(1)} <span style={{fontSize:'0.6em', color:'var(--text-muted)'}}>/ 100</span>
              </div>
            </div>
          </div>

          <div className="card" style={{ flex: '0 0 auto' }}>
            <h2>Risk Score Timeline</h2>
            <div className="chart-container">
              {scoreHistory.map((pt, i) => (
                <div 
                  key={i} 
                  className={`chart-bar ${pt.level}`} 
                  style={{ height: `${Math.max(pt.score, 2)}%` }}
                  title={`Score: ${pt.score.toFixed(1)}`}
                ></div>
              ))}
              {Array.from({ length: Math.max(0, 31 - scoreHistory.length) }).map((_, i) => (
                <div key={`empty-${i}`} className="chart-bar" style={{ opacity: 0.1 }}></div>
              ))}
            </div>
          </div>

          <div className="card" style={{ flex: 1, padding: 0, overflow: 'hidden' }}>
            <h2 style={{ padding: '1.5rem 1.5rem 0' }}>Live Event Feed</h2>
            <div className="terminal">
              {logs.map(log => (
                 <div key={log.id} className="log-line">
                   <span className="log-time">[{new Date(log.timestamp).toLocaleTimeString()}]</span>
                   <span className={`log-level ${log.level}`}>[{log.level}]</span>
                   <span>{log.raw_text}</span>
                 </div>
              ))}
              <div ref={bottomRef} />
            </div>
          </div>
        </div>

        {/* Right Column */}
        <div className="card alerts-card" style={{ gridRow: '1 / span 2' }}>
          <h2>Security Alerts</h2>
          <div className="alerts-list">
            {alerts.length === 0 ? (
               <div style={{ color: 'var(--text-muted)', textAlign: 'center', marginTop: '2rem' }}>
                 No recent alerts.
               </div>
            ) : alerts.map(alert => {
              let logsList = [];
              try {
                if (alert.triggering_logs) logsList = JSON.parse(alert.triggering_logs).slice(-3);
              } catch(e) {}
              
              return (
              <div key={alert.id} className={`alert-item ${alert.alert_level}`}>
                <div className="alert-header">
                  <span className={`alert-level ${alert.alert_level}`}>{alert.alert_level} Threat</span>
                  <span color="var(--text-muted)">{new Date(alert.timestamp).toLocaleTimeString()}</span>
                </div>
                <div className="alert-score">Risk Score: {alert.risk_score.toFixed(2)}</div>
                <div className="alert-logs">
                  {logsList.length > 0 ? logsList.map((l: string, i: number) => <div key={i}>{l.substring(0, 60)}{l.length > 60 && '...'}</div>) : "No log attached"}
                </div>
              </div>
            )})}
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
