import axios from 'axios';

const API = axios.create({
  baseURL: 'http://localhost:4000/api'
});

// trigger Python pipeline + DB update
export const runPipeline = ({ ticker, period, horizon, useLstm }) =>
  API.get('/run', {
    params: { ticker, period, horizon, use_lstm: useLstm }
  });

// fetch stored predictions
export const fetchPredictions = (ticker, model='ensemble') =>
  API.get('/predictions', { params: { ticker, model } })
     .then(res => res.data);
