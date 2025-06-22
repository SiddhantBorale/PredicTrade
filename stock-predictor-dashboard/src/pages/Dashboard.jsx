import PredictionChart from "/src/components/PredictionChart.jsx";
import PredictionTable from "/src/components/PredictionTable.jsx";

export default function Dashboard() {
  return (
    <div className="dashboard">
      <h1>📈 Stock Prediction Dashboard</h1>
      <PredictionChart />
      <PredictionTable />
    </div>
  );
}