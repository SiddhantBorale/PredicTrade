// server/index.js
import express from 'express';
import cors from 'cors';

const app = express();
const PORT = 5000;

app.use(cors());
app.use(express.json());

app.get(`https://animated-bassoon-rwqwj6rp597h5vw6-5000.app.github.dev/predictions`, (req, res) => {
  res.json([{ date: "2025-06-23", price: 123.45 }]);
});

app.listen(PORT, () => {
  console.log(`✅ Server running at http://localhost:${PORT}`);
});
