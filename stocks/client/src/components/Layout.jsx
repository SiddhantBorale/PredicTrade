// client/src/components/Layout.jsx
import React from 'react';
import { AppBar, Box, Toolbar, Typography, IconButton, Stack, Button } from '@mui/material';
import TimelineIcon from '@mui/icons-material/Timeline';
import RocketLaunchIcon from '@mui/icons-material/RocketLaunch';

export default function Layout({ onRun, children }) {
  return (
    <Box sx={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      <AppBar
        elevation={0}
        color="transparent"
        position="sticky"
        sx={{ borderBottom: '1px solid rgba(255,255,255,0.06)', backdropFilter: 'blur(8px)' }}
      >
        <Toolbar>
          <TimelineIcon sx={{ mr: 1 }} />
          <Typography variant="h6" sx={{ flexGrow: 1, fontWeight: 700 }}>
            PredicTrade
          </Typography>
          <Stack direction="row" spacing={1}>
            <IconButton color="inherit" href="https://github.com/SiddhantBorale/PredicTrade" target="_blank" size="large">
              <svg width="22" height="22" viewBox="0 0 24 24" fill="currentColor" aria-hidden>
                <path d="M12 .5C5.73.5.77 5.46.77 11.73c0 4.9 3.18 9.06 7.59 10.53.55.1.75-.23.75-.52v-1.83c-3.09.67-3.74-1.32-3.74-1.32-.5-1.27-1.22-1.6-1.22-1.6-.99-.67.08-.65.08-.65 1.1.08 1.68 1.13 1.68 1.13.98 1.68 2.58 1.2 3.21.92.1-.71.38-1.2.68-1.48-2.47-.28-5.06-1.24-5.06-5.53 0-1.22.43-2.22 1.13-3-.11-.28-.49-1.43.11-2.97 0 0 .93-.3 3.04 1.15a10.54 10.54 0 0 1 2.77-.37c.94 0 1.89.13 2.77.37 2.11-1.45 3.04-1.15 3.04-1.15.6 1.54.22 2.69.11 2.97.7.78 1.13 1.78 1.13 3 0 4.3-2.6 5.24-5.07 5.52.39.34.72 1 .72 2.02v2.99c0 .29.2.63.76.52 4.4-1.47 7.58-5.64 7.58-10.53C23.24 5.46 18.28.5 12 .5z"/>
              </svg>
            </IconButton>
          </Stack>
        </Toolbar>
      </AppBar>

      <Box sx={{ py: { xs: 5, md: 8 }, textAlign: 'center' }}>
        <Typography variant="h3" sx={{ fontWeight: 800, letterSpacing: -0.5 }}>
          PredicTrade - A ML Stock Prediction Dashboard
        </Typography>
        <Typography sx={{ opacity: 0.85, mt: 1.5 }}>
          Blend XGBoost • SARIMAX • LSTM • Ensemble
        </Typography>
      </Box>

      <Box component="main" sx={{ flex: 1, py: 3 }}>{children}</Box>
    </Box>
  );
}
