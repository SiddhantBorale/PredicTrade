import axios from 'axios';

const api = axios.create({
  // In dev, CRA proxy sends /api → http://localhost:4000
  baseURL: '/api',
  timeout: 30000
});

export async function fetchPredictions(ticker, model = 'ensemble') {
  const { data } = await api.get('/predictions', { params: { ticker, model } });
  return data; // [{date, value}]
}

export async function runPipeline({ ticker, period, horizon, useLstm }) {
  const { data } = await api.get('/run', {
    params: {
      ticker,
      period,
      horizon,
      use_lstm: !!useLstm
    }
  });
  return data;
}
