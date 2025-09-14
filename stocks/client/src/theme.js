import { createTheme } from '@mui/material/styles';

const theme = createTheme({
  palette: {
    mode: 'dark',
    background: { default: '#0b0d13', paper: '#0e1118' },
    primary: { main: '#7c5cff' },
    secondary: { main: '#22d3ee' }
  },
  shape: { borderRadius: 12 },
  components: {
    MuiPaper: { styleOverrides: { root: { border: '1px solid rgba(255,255,255,0.06)' } } }
  },
  typography: {
    fontFamily: `'Inter', ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial`
  }
});

export default theme;
