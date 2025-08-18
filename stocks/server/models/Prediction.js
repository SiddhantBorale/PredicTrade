const mongoose = require('mongoose');

const PredictionSchema = new mongoose.Schema(
  {
    ticker: { type: String, index: true, required: true, uppercase: true },
    date:   { type: Date, index: true, required: true },
    model:  { type: String, index: true, required: true, enum: ['ensemble', 'lstm', 'xgb', 'sarimax'] },
    value:  { type: Number, required: true }
  },
  { timestamps: true }
);

PredictionSchema.index({ ticker: 1, model: 1, date: 1 }, { unique: true });

module.exports = mongoose.model('Prediction', PredictionSchema);
