import React, { useMemo } from 'react';
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, Legend, CartesianGrid } from 'recharts';

// series: { ensemble: [{date,value}], lstm: [...], xgb: [...], sarimax: [...] }
function mergeSeries(series) {
  const map = new Map();
  const add = (arr, key) => {
    if (!arr) return;
    arr.forEach(({ date, value }) => {
      const d = String(date).slice(0, 10);
      if (!map.has(d)) map.set(d, { date: d });
      map.get(d)[key] = Number(value);
    });
  };
  add(series.ensemble, 'ensemble');
  add(series.lstm, 'lstm');
  add(series.xgb, 'xgb');
  add(series.sarimax, 'sarimax');
  return Array.from(map.values()).sort((a, b) => a.date.localeCompare(b.date));
}

export default function ForecastChart({ series }) {
  const data = useMemo(() => mergeSeries(series), [series]);
  const hasAny = data.length > 0;

  if (!hasAny) return <div style={{ opacity: 0.7 }}>No predictions to display yet.</div>;

  return (
    <ResponsiveContainer width="100%" height={360}>
      <LineChart data={data} margin={{ top: 8, right: 24, left: 0, bottom: 8 }}>
        <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
        <XAxis dataKey="date" />
        <YAxis />
        <Tooltip />
        <Legend />
        {'ensemble' in data[0] && <Line type="monotone" dataKey="ensemble" stroke="#7c5cff" dot={false} />}
        {'lstm' in data[0] && <Line type="monotone" dataKey="lstm" stroke="#22d3ee" dot={false} />}
        {'xgb' in data[0] && <Line type="monotone" dataKey="xgb" stroke="#34d399" dot={false} />}
        {'sarimax' in data[0] && <Line type="monotone" dataKey="sarimax" stroke="#f59e0b" dot={false} />}
      </LineChart>
    </ResponsiveContainer>
  );
}
