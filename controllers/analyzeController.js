const { spawn } = require("child_process");
const path = require("path");

exports.analyzeText = (req, res) => {
  const { text } = req.body;

  if (!text) {
    return res.status(400).json({ error: "Text missing" });
  }

  const scriptPath = path.join(__dirname, "..", "ml_service", "predict.py");

  const py = spawn("python", [scriptPath, text], {
    cwd: path.join(__dirname, "..", "ml_service")
  });

  let result = "";

  py.stdout.on("data", (data) => {
    result += data.toString();
  });

  py.stderr.on("data", (data) => {
    console.error("Python stderr:", data.toString());
  });

  py.on("close", () => {
    try {
      const json = JSON.parse(result);
      res.json(json);
    } catch (err) {
      console.error("JSON parse error:", result);
      res.status(500).json({ error: "Invalid ML response" });
    }
  });
};