require('dotenv').config();

const express = require('express');
const cors = require('cors');
const morgan = require('morgan');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');
const csv = require('csv-parser');
const mongoose = require('mongoose');

const Prediction = require('./models/Prediction');

const app = express();
app.use(cors());
app.use(express.json());
app.use(morgan('dev'));

const PORT = process.env.PORT || 4000;
const ROOT_DIR = path.resolve(__dirname, '..');        // repo root
const RESULTS_DIR = path.join(ROOT_DIR, 'results');    // Python writes CSVs here
const PYTHON = process.env.PYTHON_CMD || 'python';     // python or python3
const PY_WRITES_TO_MONGO = String(process.env.PY_WRITES_TO_MONGO || 'false').toLowerCase() === 'true';

// ────────────────────────────────────────────────────────────────
// Mongo connection
mongoose
  .connect(process.env.MONGO_URI)
  .then(() => console.log('✅ MongoDB connected'))
  .catch(err => {
    console.error('❌ Mongo connect error:', err.message);
    process.exit(1);
  });

// ────────────────────────────────────────────────────────────────
// Helpers
function safeFilePart(s) {
  // replace anything not [A-Za-z0-9_-] with _
  return String(s).replace(/[^A-Za-z0-9_-]/g, '_');
}

function readCsv(filePath) {
  return new Promise((resolve, reject) => {
    const rows = [];
    fs.createReadStream(filePath)
      .pipe(csv())
      .on('data', (row) => rows.push(row))
      .on('end', () => resolve(rows))
      .on('error', reject);
  });
}

async function importForecastCsv({ ticker, horizon, model = 'ensemble' }) {
  const safeTicker = safeFilePart(ticker);
  const file = path.join(RESULTS_DIR, `${safeTicker}_${model}_${horizon}d.csv`);

  // Wait up to ~3s for the file to appear (race-condition safe)
  for (let i = 0; i < 6; i++) {
    if (fs.existsSync(file)) break;
    await new Promise(r => setTimeout(r, 500));
  }
  if (!fs.existsSync(file)) {
    throw new Error(`Results file not found: ${file}`);
  }

  const rows = await readCsv(file);
  if (!rows.length) throw new Error(`Results file is empty: ${file}`);

  // Expected columns for your CSVs: date, forecast_close
  const ops = rows.map(r => {
    const d = new Date(r.date);
    const val = parseFloat(r.forecast_close);
    if (!isFinite(val) || isNaN(d.getTime())) return null;
    return {
      updateOne: {
        filter: { ticker: ticker.toUpperCase(), date: d, model },
        update: { $set: { value: val } },
        upsert: true
      }
    };
  }).filter(Boolean);

  if (ops.length) {
    const result = await Prediction.bulkWrite(ops, { ordered: false });
    return {
      matched: result.matchedCount ?? 0,
      upserted: result.upsertedCount ?? (result.upsertedIds ? Object.keys(result.upsertedIds).length : 0)
    };
  }
  return { matched: 0, upserted: 0 };
}

// ────────────────────────────────────────────────────────────────
// Routes

app.get('/api/health', (req, res) => {
  res.json({ ok: true, time: new Date().toISOString() });
});

// GET /api/predictions?ticker=PLTR&model=ensemble
app.get('/api/predictions', async (req, res) => {
  try {
    const { ticker, model = 'ensemble' } = req.query;
    if (!ticker) return res.status(400).json({ error: 'ticker is required' });

    const docs = await Prediction
      .find({ ticker: ticker.toUpperCase(), model })
      .sort({ date: 1 })
      .lean();

    if (!docs.length) {
      return res.status(404).json({ error: `No predictions found for ${ticker} (${model}).` });
    }

    res.json(docs.map(d => ({
      date: d.date.toISOString().slice(0, 10),
      value: d.value
    })));
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// GET /api/run?ticker=PLTR&period=6mo&horizon=7&use_lstm=true
app.get('/api/run', async (req, res) => {
  const { ticker = 'PLTR', period = '6mo', horizon = 7, use_lstm = 'false' } = req.query;

  // Build args for your existing Python orchestrator
  const args = ['src/main.py', '--ticker', ticker, '--period', String(period), '--horizon', String(horizon)];
  if (String(use_lstm) === 'true') args.push('--use_lstm');

  const logs = [];
  const child = spawn(PYTHON, args, { cwd: ROOT_DIR });

  child.stdout.on('data', (d) => {
    const s = d.toString();
    logs.push(s);
    process.stdout.write(s);
  });
  child.stderr.on('data', (d) => {
    const s = d.toString();
    logs.push(s);
    process.stderr.write(s);
  });

  child.on('error', (err) => {
    return res.status(500).json({ error: `Failed to start Python: ${err.message}`, logs: logs.join('') });
  });

  child.on('close', async (code) => {
    if (code !== 0) {
      return res.status(500).json({
        error: `Python exited with code ${code}`,
        logs: logs.join('')
      });
    }

    // If Python writes to Mongo itself, we can stop here
    if (PY_WRITES_TO_MONGO) {
      return res.json({ message: 'Pipeline completed (Python wrote directly to Mongo).', logs: logs.join('') });
    }

    // Otherwise, import CSVs produced by Python
    try {
      // Import ENSEMBLE by default; add others if you produce those CSVs
      const imported = {};
      imported.ensemble = await importForecastCsv({ ticker, horizon, model: 'ensemble' }).catch(e => ({ error: e.message }));

      // If you also save SARIMAX/LSTM/XGB CSVs, uncomment:
      // imported.sarimax = await importForecastCsv({ ticker, horizon, model: 'sarimax' }).catch(e => ({ error: e.message }));
      // imported.lstm    = await importForecastCsv({ ticker, horizon, model: 'lstm' }).catch(e => ({ error: e.message }));
      // imported.xgb     = await importForecastCsv({ ticker, horizon, model: 'xgb' }).catch(e => ({ error: e.message }));

      res.json({ message: 'Pipeline completed', imported, logs: logs.join('') });
    } catch (e) {
      res.status(500).json({
        error: `Pipeline done, but importing CSVs failed: ${e.message}`,
        logs: logs.join('')
      });
    }
  });
});

// ────────────────────────────────────────────────────────────────
app.listen(PORT, () => {
  console.log(`🚀 Server listening on http://localhost:${PORT}`);
  console.log(`Root dir: ${ROOT_DIR}`);
  console.log(`Results dir: ${RESULTS_DIR}`);
});
