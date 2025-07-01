import React, { useEffect, useState } from 'react';
import { fetchPredictions } from '../api';

export default function PredictionsList({ ticker }) {
  const [preds, setPreds] = useState([]);

  useEffect(() => {
    fetchPredictions(ticker).then(setPreds);
  }, [ticker]);

  return (
    <table>
      <thead>
        <tr><th>Date</th><th>Forecast</th></tr>
      </thead>
      <tbody>
        {preds.map((p, i) => (
          <tr key={i}>
            <td>{p.date}</td>
            <td>{p.value.toFixed(2)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
