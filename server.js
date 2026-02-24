const express = require("express");
const cors = require("cors");

const analyzeRoutes = require("./routes/analyze");

const app = express();
const PORT = process.env.PORT || 5000;

app.use(cors({
  origin: [
    "https://linguistic-based-fake-news-detection.netlify.app"
  ],
  methods: ["GET", "POST"],
  allowedHeaders: ["Content-Type"]
}));

app.use(express.json());

app.use("/api/analyze", analyzeRoutes);

app.get("/", (req, res) => {
  res.send("Fake News Detection API Running");
});

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});