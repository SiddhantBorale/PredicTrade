import { useEffect, useState } from "react";
import axios from "axios";

export default function PredictionTable() {
  const [predictions, setPredictions] = useState([]);

  useEffect(() => {
    axios.get("https://<your-codespace-name>-5000.app.github.dev/predictions")
      .then((res) => {
        setPredictions(res.data.reverse()); // newest first
      })
      .catch((err) => console.error("Error fetching predictions:", err));
  }, []);

  return (
    <div className="prediction-table">
      <h2>📋 Prediction History</h2>
      <table>
        <thead>
          <tr>
            <th>Date</th>
            <th>Ticker</th>
            <th>Prediction</th>
            <th>Confidence</th>
            <th>Actual Result</th>
            <th>Price @ Prediction</th>
            <th>Actual Price</th>
          </tr>
        </thead>
        <tbody>
          {predictions.map((p, i) => (
            <tr key={i}>
              <td>{p.date}</td>
              <td>{p.ticker}</td>
              <td>{p.prediction}</td>
              <td>{p.confidence}</td>
              <td>{p.actual_result || "–"}</td>
              <td>{p.close_price_at_prediction}</td>
              <td>{p.actual_price || "–"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}