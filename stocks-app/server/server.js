require('dotenv').config();
const express    = require('express');
const mongoose   = require('mongoose');
const cors       = require('cors');
const { exec }   = require('child_process');
const Prediction = require('./models/Prediction');
const path       = require('path');

const app = express();
app.use(cors());
app.use(express.json());

const PORT      = process.env.PORT || 4000;
const PYTHON    = process.env.PYTHON_CMD || 'python';
const ROOT_DIR  = path.resolve(__dirname, '..');
const RESULTS   = path.join(ROOT_DIR, 'results');

// Connect to MongoDB
mongoose
  .connect(process.env.MONGO_URI)
  .then(() => console.log('✅ MongoDB connected'))
  .catch(err => console.error('Mongo connect error:', err));

// ── Route: trigger Python pipeline ─────────────────────────────
// GET /api/run?ticker=PLTR&period=6mo&horizon=7&use_lstm=true
app.get('/api/run', (req, res) => {
  const { ticker='PLTR', period='6mo', horizon=7, use_lstm='false' } = req.query;
  const lstmFlag = use_lstm==='true' ? '--use_lstm' : '';
  const cmd = `${PYTHON} src/main.py --ticker ${ticker} --period ${period} --horizon ${horizon} ${lstmFlag}`;

  exec(cmd, { cwd: ROOT_DIR }, (err, stdout, stderr) => {
    if (err) {
      console.error('Pipeline error:', stderr);
      return res.status(500).json({ error: stderr || err.message });
    }
    console.log('Pipeline output:', stdout);
    // After Python run, load the ensemble CSV into Mongo
    const csvFile = path.join(RESULTS, `${ticker}_ensemble_${horizon}d.csv`);
    const fs = require('fs');
    const csv = require('csv-parser');
    const bulk = [];

    fs.createReadStream(csvFile)
      .pipe(csv())
      .on('data', row => {
        bulk.push({
          updateOne: {
            filter: { ticker, date: new Date(row.date), model: 'ensemble' },
            update: { $set: { value: parseFloat(row.forecast_close) } },
            upsert: true
          }
        });
      })
      .on('end', async () => {
        if (bulk.length) {
          const result = await Prediction.bulkWrite(bulk);
          console.log(`Mongo upsert: matched=${result.matchedCount}, upserted=${result.upsertedCount}`);
        }
        res.json({ message: 'Pipeline run and DB updated' });
      });
  });
});

// ── Route: get predictions from Mongo ───────────────────────────
// GET /api/predictions?ticker=PLTR&model=ensemble
app.get('/api/predictions', async (req, res) => {
  try {
    const { ticker, model='ensemble' } = req.query;
    const filter = {};
    if (ticker) filter.ticker = ticker.toUpperCase();
    if (model)  filter.model  = model;
    const preds = await Prediction.find(filter).sort({ date: 1 }).lean();
    res.json(preds.map(p => ({
      date:  p.date.toISOString().slice(0,10),
      value: p.value
    })));
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

app.listen(PORT, () =>
  console.log(`🚀 Server listening: http://localhost:${PORT}`)
);
