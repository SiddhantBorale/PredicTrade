import React, { useState } from 'react';
import { runPipeline } from '../api';

export default function RunButton({ ticker, period, horizon, onDone }) {
  const [loading, setLoading] = useState(false);
  const handleClick = () => {
    setLoading(true);
    runPipeline({ ticker, period, horizon, useLstm: true })
      .then(() => onDone())
      .finally(() => setLoading(false));
  };
  return (
    <button onClick={handleClick} disabled={loading}>
      {loading ? 'Running…' : 'Refresh Forecast'}
    </button>
  );
}
