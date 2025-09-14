// client/src/components/Controls.jsx
import React from 'react';
import {
  Paper,
  Grid,
  TextField,
  MenuItem,
  Slider,
  Box,
  FormControlLabel,
  Switch,
  Button,
  Tooltip,
  Typography,
} from '@mui/material';

const PERIODS = ['1mo', '2mo', '3mo', '6mo', '9mo', '1y', '2y'];

export default function Controls({
  ticker, setTicker,
  period, setPeriod,
  horizon, setHorizon,
  models, setModels,
  onRun
}) {
  const handleToggle = (key) => (_, val) =>
    setModels((m) => ({ ...m, [key]: val }));

  return (
    <Paper sx={{ p: 2, mb: 2 }}>
      <Grid container spacing={2} alignItems="center">
        {/* Ticker */}
        <Grid item xs={12} md={3}>
          <TextField
            label="Ticker"
            fullWidth
            value={ticker}
            onChange={(e) => setTicker(e.target.value.toUpperCase())}
            placeholder="AAPL"
          />
        </Grid>

        {/* Period */}
        <Grid item xs={12} md={3}>
          <TextField
            select
            fullWidth
            label="Period"
            value={period}
            onChange={(e) => setPeriod(e.target.value)}
          >
            {PERIODS.map((p) => (
              <MenuItem key={p} value={p}>{p}</MenuItem>
            ))}
          </TextField>
        </Grid>

        {/* Horizon */}
        <Grid item xs={12} md={3}>
          <Typography variant="caption" sx={{ display: 'block', mb: 0.5 }}>
            Forecast Horizon (days)
          </Typography>
          <Slider
            value={horizon}
            min={1}
            max={14}
            step={1}
            onChange={(_, v) => setHorizon(v)}
            valueLabelDisplay="auto"
          />
        </Grid>

        {/* Run button */}
        <Grid item xs={12} md={3} sx={{ textAlign: { xs: 'left', md: 'right' } }}>
          <Button onClick={onRun} variant="contained" size="large">
            Run
          </Button>
        </Grid>

        {/* Model toggles */}
        <Grid item xs={12}>
          <Box
            sx={{
              display: 'flex',
              flexWrap: 'wrap',
              gap: 2.5,
              alignItems: 'center',
              mt: 0.5,
            }}
          >
            <Tooltip title="Blended forecast saved as 'ensemble'">
              <FormControlLabel
                control={
                  <Switch
                    color="primary"
                    checked={!!models.ensemble}
                    onChange={handleToggle('ensemble')}
                  />
                }
                label="Ensemble"
              />
            </Tooltip>

            <Tooltip title="Sequence model (requires TensorFlow)">
              <FormControlLabel
                control={
                  <Switch
                    color="primary"
                    checked={!!models.lstm}
                    onChange={handleToggle('lstm')}
                  />
                }
                label="LSTM"
              />
            </Tooltip>

            <Tooltip title="Tree-boosted regressor over engineered features">
              <FormControlLabel
                control={
                  <Switch
                    color="primary"
                    checked={!!models.xgb}
                    onChange={handleToggle('xgb')}
                  />
                }
                label="XGBoost"
              />
            </Tooltip>

            <Tooltip title="Time-series model with seasonality/exog">
              <FormControlLabel
                control={
                  <Switch
                    color="primary"
                    checked={!!models.sarimax}
                    onChange={handleToggle('sarimax')}
                  />
                }
                label="SARIMAX"
              />
            </Tooltip>
          </Box>
        </Grid>
      </Grid>
    </Paper>
  );
}
