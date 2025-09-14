import axios from 'axios';

const api = axios.create({
  // proxy in package.json points to http://localhost:4000
  timeout: 120000
});

export async function runPipeline({ ticker, period, horizon, useLstm }) {
  const params = {
    ticker,
    period,
    horizon,
    use_lstm: !!useLstm
  };
  const { data } = await api.get('/api/run', { params });
  return data;
}

export async function fetchPredictions(ticker, model = 'ensemble') {
  const { data } = await api.get('/api/predictions', { params: { ticker, model } });
  // data -> [{ date: 'YYYY-MM-DD', value: number }]
  return data;
}
