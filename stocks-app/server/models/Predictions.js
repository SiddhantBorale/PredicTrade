const mongoose = require('mongoose');

const predictionSchema = new mongoose.Schema({
  ticker:    { type: String, required: true },
  date:      { type: Date,   required: true },
  model:     { type: String, required: true }, // 'ensemble' or 'lstm', etc.
  value:     { type: Number, required: true },
  createdAt: { type: Date,   default: Date.now }
});

module.exports = mongoose.model('Prediction', predictionSchema);
