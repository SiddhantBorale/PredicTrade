import PredictionChart from './components/PredictionChart';
import './App.css'; // or use index.css if preferred

export default function Dashboard() {
  return (
    <div className="dashboard">
      <h1>📈 Stock Prediction Dashboard</h1>
      <PredictionChart />
    </div>
  );
}