// server/server.js
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

// ────────────────────────────────────────────────────────────────
// Config
const PORT = process.env.PORT || 4000;
const ROOT_DIR = path.resolve(__dirname, '..'); // repo root
const RESULTS_PRIMARY = path.join(ROOT_DIR, 'results'); // Python writes here
const RESULTS_FALLBACK = path.join(ROOT_DIR, 'results'); // older path, just in case
const RESULTS_DIRS = [RESULTS_PRIMARY, RESULTS_FALLBACK];

const PYTHON = process.env.PYTHON_CMD || 'python3';
const PY_WRITES_TO_MONGO = String(process.env.PY_WRITES_TO_MONGO || 'false').toLowerCase() === 'true';

// ────────────────────────────────────────────────────────────────
// Mongo connection (with in-memory fallback for Codespaces)
let memServer = null;
async function connectMongo() {
  const uri = process.env.MONGO_URI;
  try {
    if (!uri) throw new Error('MONGO_URI not provided');
    await mongoose.connect(uri);
    console.log('✅ MongoDB connected:', uri);
  } catch (e) {
    console.warn('⚠️ Mongo connect failed, using in-memory Mongo:', e.message);
    try {
      const { MongoMemoryServer } = require('mongodb-memory-server');
      memServer = await MongoMemoryServer.create();
      const memUri = memServer.getUri();
      await mongoose.connect(memUri);
      console.log('✅ In-memory MongoDB started');
    } catch (err) {
      console.error('❌ Could not start in-memory MongoDB:', err.message);
      console.error('   API will still run; /api/predictions will fall back to CSV.');
      // No process.exit here — keep the API alive
    }
  }
}
connectMongo().catch((e) => {
  console.error('❌ Unexpected Mongo init error:', e);
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

function findForecastCsv({ ticker, horizon, model }) {
  const safeTicker = safeFilePart(ticker);
  const filename = `${safeTicker}_${model}_${horizon}d.csv`;
  for (const dir of RESULTS_DIRS) {
    const file = path.join(dir, filename);
    if (fs.existsSync(file)) return file;
  }
  return null;
}

async function importForecastCsv({ ticker, horizon, model = 'ensemble' }) {
  // wait briefly for file to appear (race protection)
  let file = null;
  for (let i = 0; i < 6; i++) {
    file = findForecastCsv({ ticker, horizon, model });
    if (file) break;
    await new Promise((r) => setTimeout(r, 500));
  }
  if (!file) throw new Error(`Results file not found for ${model} horizon=${horizon}`);

  const rows = await readCsv(file);
  if (!rows.length) throw new Error(`Results file is empty: ${file}`);

  // expected columns: date, forecast_close
  const ops = rows
    .map((r) => {
      const d = new Date(r.date);
      const val = parseFloat(r.forecast_close);
      if (!isFinite(val) || isNaN(d.getTime())) return null;
      return {
        updateOne: {
          filter: { ticker: ticker.toUpperCase(), date: d, model },
          update: { $set: { value: val } },
          upsert: true,
        },
      };
    })
    .filter(Boolean);

  if (ops.length) {
    const result = await Prediction.bulkWrite(ops, { ordered: false });
    return {
      matched: result.matchedCount ?? 0,
      upserted:
        result.upsertedCount ??
        (result.upsertedIds ? Object.keys(result.upsertedIds).length : 0),
      file,
    };
  }
  return { matched: 0, upserted: 0, file };
}

// ────────────────────────────────────────────────────────────────
// Routes
app.get('/api/health', (req, res) => {
  res.json({ ok: true, time: new Date().toISOString() });
});

// GET /api/predictions?ticker=PLTR&model=ensemble&horizon=7
// Returns DB data; falls back to CSV if DB empty/unavailable
app.get('/api/predictions', async (req, res) => {
  const { ticker, model = 'ensemble' } = req.query;
  const horizon = parseInt(req.query.horizon || '7', 10);

  if (!ticker) return res.status(400).json({ error: 'ticker is required' });

  try {
    // If Mongo is connected, try DB first
    if (mongoose.connection.readyState === 1) {
      const docs = await Prediction.find({
        ticker: ticker.toUpperCase(),
        model,
      })
        .sort({ date: 1 })
        .lean();

      if (docs && docs.length) {
        return res.json(
          docs.map((d) => ({
            date: d.date.toISOString().slice(0, 10),
            value: d.value,
          }))
        );
      }
    }

    // CSV fallback
    const file = findForecastCsv({ ticker, horizon, model });
    if (!file) {
      return res
        .status(404)
        .json({ error: `No predictions found for ${ticker} (${model}).` });
    }
    const rows = await readCsv(file);
    if (!rows.length)
      return res.status(404).json({ error: `Empty CSV for ${ticker} (${model}).` });

    return res.json(
      rows.map((r) => ({
        date: String(r.date).slice(0, 10),
        value: Number(r.forecast_close),
      }))
    );
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// GET /api/run?ticker=PLTR&period=6mo&horizon=7&use_lstm=true
app.get('/api/run', async (req, res) => {
  const { ticker = 'PLTR', period = '6mo' } = req.query;
  const horizon = parseInt(req.query.horizon || '7', 10);
  const use_lstm = String(req.query.use_lstm || 'false') === 'true';

  const args = [
    'src/main.py',
    '--ticker',
    ticker,
    '--period',
    String(period),
    '--horizon',
    String(horizon),
  ];
  if (use_lstm) args.push('--use_lstm');

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
    return res
      .status(500)
      .json({ error: `Failed to start Python: ${err.message}`, logs: logs.join('') });
  });

  child.on('close', async (code) => {
    if (code !== 0) {
      return res.status(500).json({
        error: `Python exited with code ${code}`,
        logs: logs.join(''),
      });
    }

    if (PY_WRITES_TO_MONGO) {
      return res.json({
        message: 'Pipeline completed (Python wrote directly to Mongo).',
        logs: logs.join(''),
      });
    }

    // Import whatever models exist without failing the whole request
    const imported = {};
    const models = ['ensemble', 'sarimax', 'lstm', 'xgb'];
    for (const m of models) {
      imported[m] = await importForecastCsv({ ticker, horizon, model: m }).catch(
        (e) => ({ error: e.message })
      );
    }

    res.json({ message: 'Pipeline completed', imported, logs: logs.join('') });
  });
});

// ────────────────────────────────────────────────────────────────
app.listen(PORT, () => {
  console.log(`🚀 Server listening on http://localhost:${PORT}`);
  console.log(`Root dir:     ${ROOT_DIR}`);
  console.log(`Results dirs: ${RESULTS_DIRS.join(' , ')}`);
});
