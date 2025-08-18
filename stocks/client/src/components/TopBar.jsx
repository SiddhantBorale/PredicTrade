import * as React from 'react';
import { AppBar, Toolbar, Typography } from '@mui/material';
import ShowChartIcon from '@mui/icons-material/ShowChart';

export default function TopBar() {
  return (
    <AppBar position="static" color="primary" enableColorOnDark>
      <Toolbar>
        <ShowChartIcon sx={{ mr: 1 }} />
        <Typography variant="h6" component="div">
          Stocks Forecast Dashboard
        </Typography>
      </Toolbar>
    </AppBar>
  );
}
