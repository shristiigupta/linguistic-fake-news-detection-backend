const { spawn } = require("child_process");
const path = require("path");

exports.analyzeText = (req, res) => {
  const { text } = req.body;
  if (!text) return res.status(400).json({ error: "Text missing" });

  // Get the directory where the script is located
  const scriptPath = path.join(__dirname, "..", "ml_service", "predict.py");
  
  // Pass text as command line argument and run from ml_service directory
  const py = spawn("python", [scriptPath, text], {
    cwd: path.join(__dirname, "..", "ml_service")
  });

  let result = "";
  py.stdout.on("data", (data) => {
    result += data.toString();
  });

  py.stderr.on("data", (data) => {
    console.error("Python error:", data.toString());
  });

  py.on("close", (code) => {
    if (code !== 0) {
      console.error("Python process exited with code:", code);
      return res.status(500).json({ error: "ML processing failed" });
    }
    try {
      const json = JSON.parse(result);
      res.json(json);
    } catch (e) {
      console.error("Parsing error:", e, "Result was:", result);
      res.status(500).json({ error: "ML processing failed" });
    }
  });
};
