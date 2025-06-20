import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts';

const sampleData = [
  { week: 'May 1', prediction: 152 },
  { week: 'May 8', prediction: 158 },
  { week: 'May 15', prediction: 155 },
  { week: 'May 22', prediction: 162 },
  { week: 'May 29', prediction: 167 },
];

export default function PredictionChart({ data = sampleData }) {
  return (
    <div className="chart-container">
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="week" />
          <YAxis />
          <Tooltip />
          <Line type="monotone" dataKey="prediction" stroke="#8884d8" />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}