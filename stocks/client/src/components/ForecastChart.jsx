import * as React from 'react';
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, Legend, CartesianGrid } from 'recharts';
import dayjs from 'dayjs';

function mergeSeries(seriesMap) {
  // seriesMap: { ensemble: [{date,value}], lstm: [...], ... }
  const allDates = new Set();
  Object.values(seriesMap).forEach(arr => (arr || []).forEach(p => allDates.add(p.date)));
  const dates = Array.from(allDates).sort();
  return dates.map(d => {
    const row = { date: d };
    Object.keys(seriesMap).forEach(k => {
      const found = (seriesMap[k] || []).find(x => x.date === d);
      if (found) row[k] = found.value;
    });
    return row;
  });
}

export default function ForecastChart({ series }) {
  const data = React.useMemo(() => mergeSeries(series), [series]);

  return (
    <ResponsiveContainer width="100%" height={360}>
      <LineChart data={data} margin={{ top: 10, right: 20, bottom: 10, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis
          dataKey="date"
          tickFormatter={(d) => dayjs(d).format('MMM D')}
          minTickGap={24}
        />
        <YAxis domain={['auto', 'auto']} />
        <Tooltip
          labelFormatter={(d) => dayjs(d).format('YYYY-MM-DD')}
          formatter={(val, name) => [val?.toFixed?.(2), name.toUpperCase()]}
        />
        <Legend />
        {/* Lines auto-colored by Recharts; names must match keys in `data` */}
        {series.ensemble && <Line type="monotone" dataKey="ensemble" dot={false} strokeWidth={2} />}
        {series.lstm &&     <Line type="monotone" dataKey="lstm"     dot={false} strokeWidth={2} />}
        {series.xgb &&      <Line type="monotone" dataKey="xgb"      dot={false} strokeWidth={2} />}
        {series.sarimax &&  <Line type="monotone" dataKey="sarimax"  dot={false} strokeWidth={2} />}
      </LineChart>
    </ResponsiveContainer>
  );
}
