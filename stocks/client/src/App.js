import * as React from 'react';
import { Container, Grid, Paper, Typography, Alert, CircularProgress, Snackbar } from '@mui/material';
import TopBar from './components/TopBar';
import Controls from './components/Controls';
import ForecastChart from './components/ForecastChart';
import PredictionsTable from './components/PredictionsTable';
import { fetchPredictions, runPipeline } from './api';

export default function App() {
  const [ticker, setTicker] = React.useState(localStorage.getItem('ticker') || 'PLTR');
  const [period, setPeriod] = React.useState(localStorage.getItem('period') || '6mo');
  const [horizon, setHorizon] = React.useState(Number(localStorage.getItem('horizon') || 7));
  const [models, setModels] = React.useState({ ensemble: true, lstm: false, xgb: false, sarimax: false });

  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState('');
  const [snack, setSnack] = React.useState('');

  const [series, setSeries] = React.useState({
    ensemble: null, lstm: null, xgb: null, sarimax: null
  });

  // persist simple prefs
  React.useEffect(() => {
    localStorage.setItem('ticker', ticker);
    localStorage.setItem('period', period);
    localStorage.setItem('horizon', String(horizon));
  }, [ticker, period, horizon]);

  async function loadSeries() {
    setLoading(true);
    setError('');
    try {
      const next = { ensemble: null, lstm: null, xgb: null, sarimax: null };
      if (models.ensemble) next.ensemble = await fetchPredictions(ticker, 'ensemble');
      if (models.lstm)     next.lstm     = await fetchPredictions(ticker, 'lstm');
      if (models.xgb)      next.xgb      = await fetchPredictions(ticker, 'xgb');
      if (models.sarimax)  next.sarimax  = await fetchPredictions(ticker, 'sarimax');
      setSeries(next);
    } catch (e) {
      setError(e?.response?.data?.error || e.message);
    } finally {
      setLoading(false);
    }
  }

  React.useEffect(() => {
    loadSeries();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ticker, models.ensemble, models.lstm, models.xgb, models.sarimax]);

  async function handleRun() {
    setLoading(true);
    setError('');
    setSnack('');
    try {
      await runPipeline({ ticker, period, horizon, useLstm: models.lstm });
      setSnack('Pipeline completed. Loading latest predictions…');
      // re-fetch predictions after run
      await loadSeries();
    } catch (e) {
      setError(e?.response?.data?.error || e.message);
    } finally {
      setLoading(false);
    }
  }

  const tableRows = series.ensemble || series.lstm || series.xgb || series.sarimax || [];

  return (
    <>
      <TopBar />
      <Container maxWidth="lg" sx={{ mt: 3, mb: 6 }}>
        <Controls
          ticker={ticker} setTicker={setTicker}
          period={period} setPeriod={setPeriod}
          horizon={horizon} setHorizon={setHorizon}
          models={models} setModels={setModels}
          onRun={handleRun}
        />

        {error && <Alert severity="error" sx={{ mt: 2 }}>{error}</Alert>}

        <Grid container spacing={2} sx={{ mt: 2 }}>
          <Grid item xs={12} md={8}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>Forecast</Typography>
              {loading ? (
                <CircularProgress size={28} />
              ) : (
                <ForecastChart series={series} />
              )}
            </Paper>
          </Grid>
          <Grid item xs={12} md={4}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>Details</Typography>
              <PredictionsTable rows={tableRows} label={
                models.ensemble ? 'Ensemble'
                : models.lstm ? 'LSTM'
                : models.xgb ? 'XGBoost'
                : 'SARIMAX'
              } />
            </Paper>
          </Grid>
        </Grid>

        <Snackbar
          open={!!snack}
          autoHideDuration={3000}
          onClose={() => setSnack('')}
          message={snack}
        />
      </Container>
    </>
  );
}
