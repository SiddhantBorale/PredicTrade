import React, { useState } from 'react';
import RunButton from './components/RunButton';
import PredictionsList from './components/PredictionsList';

function App() {
  const [ticker, setTicker]   = useState('PLTR');
  const [period, setPeriod]   = useState('6mo');
  const [horizon, setHorizon] = useState(7);
  const [dirty, setDirty]     = useState(false);

  const refresh = () => setDirty(d => !d);

  return (
    <div style={{ padding: 20 }}>
      <h1>Stock Forecast Dashboard</h1>
      <div>
        <label>
          Ticker:{' '}
          <input
            value={ticker}
            onChange={e => setTicker(e.target.value.toUpperCase())}
          />
        </label>{' '}
        <label>
          Period:{' '}
          <input
            value={period}
            onChange={e => setPeriod(e.target.value)}
          />
        </label>{' '}
        <label>
          Horizon:{' '}
          <input
            type="number"
            value={horizon}
            onChange={e => setHorizon(+e.target.value)}
          />
        </label>{' '}
        <RunButton
          ticker={ticker}
          period={period}
          horizon={horizon}
          onDone={refresh}
        />
      </div>

      <h2>Ensemble Forecast for {ticker}</h2>
      <PredictionsList key={dirty} ticker={ticker} />
    </div>
  );
}

export default App;
