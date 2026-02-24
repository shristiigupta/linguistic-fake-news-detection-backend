const express = require("express");
const cors = require("cors");

const analyzeRoutes = require("./routes/analyze");

const app = express();
const PORT = 5000;

app.use(cors());
app.use(express.json()); 

app.use("/api/analyze", analyzeRoutes);

app.get("/", (req, res) => {
    res.send("Fake News Detection API Running");
});

app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
});