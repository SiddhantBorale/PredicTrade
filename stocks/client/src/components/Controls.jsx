import * as React from 'react';
import {
  Box, TextField, MenuItem, Button, FormGroup, FormControlLabel, Checkbox, Stack
} from '@mui/material';

const PERIODS = ['1mo', '3mo', '6mo', '1y', '2y'];
const HORIZONS = [1, 3, 5, 7, 14];

export default function Controls({
  ticker, setTicker,
  period, setPeriod,
  horizon, setHorizon,
  models, setModels,
  onRun
}) {
  const toggleModel = (name) => (e) => {
    setModels(m => ({ ...m, [name]: e.target.checked }));
  };

  return (
    <Box sx={{ p: 2, borderRadius: 1, bgcolor: 'background.paper' }}>
      <Stack direction={{ xs: 'column', md: 'row' }} spacing={2} alignItems="center">
        <TextField
          label="Ticker"
          value={ticker}
          onChange={e => setTicker(e.target.value.toUpperCase())}
          inputProps={{ style: { textTransform: 'uppercase' } }}
        />
        <TextField select label="Period" value={period} onChange={e => setPeriod(e.target.value)}>
          {PERIODS.map(p => <MenuItem key={p} value={p}>{p}</MenuItem>)}
        </TextField>
        <TextField select label="Horizon" value={horizon} onChange={e => setHorizon(Number(e.target.value))}>
          {HORIZONS.map(h => <MenuItem key={h} value={h}>{h} days</MenuItem>)}
        </TextField>

        <FormGroup row>
          <FormControlLabel control={
            <Checkbox checked={!!models.ensemble} onChange={toggleModel('ensemble')} />
          } label="Ensemble" />
          <FormControlLabel control={
            <Checkbox checked={!!models.lstm} onChange={toggleModel('lstm')} />
          } label="LSTM" />
          <FormControlLabel control={
            <Checkbox checked={!!models.xgb} onChange={toggleModel('xgb')} />
          } label="XGBoost" />
          <FormControlLabel control={
            <Checkbox checked={!!models.sarimax} onChange={toggleModel('sarimax')} />
          } label="SARIMAX" />
        </FormGroup>

        <Button variant="contained" onClick={onRun}>
          Run Pipeline
        </Button>
      </Stack>
    </Box>
  );
}
