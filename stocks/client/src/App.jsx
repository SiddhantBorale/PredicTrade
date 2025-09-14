// client/src/App.jsx
import React, { useCallback, useEffect, useState } from 'react';
import {
  Container, Grid, Paper, Typography, Alert, CircularProgress, Snackbar
} from '@mui/material';
import Layout from './components/Layout';
import Controls from './components/Controls';
import ForecastChart from './components/ForecastChart';
import PredictionsTable from './components/PredictionsTable';
import { fetchPredictions, runPipeline } from './api';

const MODEL_META = [
  { key: 'ensemble', label: 'Ensemble' , color: '#7c5cff' },
  { key: 'lstm',     label: 'LSTM'     , color: '#22d3ee' },
  { key: 'xgb',      label: 'XGBoost'  , color: '#34d399' },
  { key: 'sarimax',  label: 'SARIMAX'  , color: '#f59e0b' },
];

export default function App() {
  const [ticker, setTicker]   = useState(localStorage.getItem('ticker') || 'PLTR');
  const [period, setPeriod]   = useState(localStorage.getItem('period') || '6mo');
  const [horizon, setHorizon] = useState(Number(localStorage.getItem('horizon') || 7));
  const [models, setModels]   = useState(
    JSON.parse(localStorage.getItem('models') || '{"ensemble":true,"lstm":false,"xgb":false,"sarimax":false}')
  );

  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState('');
  const [snack, setSnack]     = useState('');

  const [series, setSeries] = useState({ ensemble: null, lstm: null, xgb: null, sarimax: null });

  // persist prefs
  useEffect(() => localStorage.setItem('ticker', ticker), [ticker]);
  useEffect(() => localStorage.setItem('period', period), [period]);
  useEffect(() => localStorage.setItem('horizon', String(horizon)), [horizon]);
  useEffect(() => localStorage.setItem('models', JSON.stringify(models)), [models]);

  // fetch selected series
  const loadSeries = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      const next = { ensemble: null, lstm: null, xgb: null, sarimax: null };
      const tasks = [];
      if (models.ensemble) tasks.push(fetchPredictions(ticker, 'ensemble').then(d => (next.ensemble = d)).catch(() => null));
      if (models.lstm)     tasks.push(fetchPredictions(ticker, 'lstm').then(d => (next.lstm = d)).catch(() => null));
      if (models.xgb)      tasks.push(fetchPredictions(ticker, 'xgb').then(d => (next.xgb = d)).catch(() => null));
      if (models.sarimax)  tasks.push(fetchPredictions(ticker, 'sarimax').then(d => (next.sarimax = d)).catch(() => null));

      await Promise.all(tasks);

      if (!next.ensemble && !next.lstm && !next.xgb && !next.sarimax) {
        throw new Error(`No predictions found for ${ticker}. Try running the pipeline.`);
      }
      setSeries(next);
    } catch (e) {
      setSeries({ ensemble: null, lstm: null, xgb: null, sarimax: null });
      setError(e?.response?.data?.error || e.message);
    } finally {
      setLoading(false);
    }
  }, [ticker, models.ensemble, models.lstm, models.xgb, models.sarimax]);

  useEffect(() => { loadSeries(); }, [loadSeries]);

  const handleRun = useCallback(async () => {
    setLoading(true);
    setError('');
    setSnack('');
    try {
      await runPipeline({ ticker, period, horizon, useLstm: models.lstm });
      setSnack('Pipeline completed. Loading latest predictions…');
      await loadSeries();
    } catch (e) {
      setError(e?.response?.data?.error || e.message);
    } finally {
      setLoading(false);
    }
  }, [ticker, period, horizon, models.lstm, loadSeries]);

  // which model cards to render (only selected + has data)
  const activeCards = MODEL_META.filter(m => models[m.key] && Array.isArray(series[m.key]));

  return (
    <Layout onRun={handleRun}>
      <Container maxWidth="lg">
        <Controls
          ticker={ticker} setTicker={setTicker}
          period={period} setPeriod={setPeriod}
          horizon={horizon} setHorizon={setHorizon}
          models={models} setModels={setModels}
          onRun={handleRun}
        />

        {error && <Alert severity="error" sx={{ mt: 2 }}>{error}</Alert>}

        <Grid container spacing={2} sx={{ mt: 1 }}>
          {/* Chart */}
          <Grid item xs={12}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>Forecast</Typography>
              {loading ? <CircularProgress size={28} /> : <ForecastChart series={series} />}
            </Paper>
          </Grid>

          {/* Show a table for EVERY selected model */}
          {activeCards.length === 0 ? (
            <Grid item xs={12}>
              <Paper sx={{ p: 2, opacity: 0.8 }}>
                <Typography>No model selected, or no predictions yet.</Typography>
              </Paper>
            </Grid>
          ) : (
            activeCards.map(({ key, label, color }) => (
              <Grid key={key} item xs={12} md={6} lg={3}>
                <Paper sx={{ p: 2 }}>
                  <Typography variant="h6" gutterBottom sx={{ color }}>
                    {label}
                  </Typography>
                  <PredictionsTable rows={series[key]} label={label} />
                </Paper>
              </Grid>
            ))
          )}
        </Grid>

        <Snackbar open={!!snack} autoHideDuration={3200} onClose={() => setSnack('')} message={snack} />
      </Container>
    </Layout>
  );
}
