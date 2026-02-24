const axios = require("axios");

// URL for the ML Flask service
// In production, this would be the URL of your deployed Flask service on Render
//const ML_SERVICE_URL = process.env.ML_SERVICE_URL || "http://localhost:5001";
ML_SERVICE_URL= "https://fake-news-ml.onrender.com"

exports.analyzeText = async (req, res) => {
  const { text } = req.body;

  if (!text) {
    return res.status(400).json({ error: "Text missing" });
  }

  try {
    // Call the Flask ML service
    const response = await axios.post(`${ML_SERVICE_URL}/predict`, {
      text: text
    });

    res.json(response.data);
  } catch (error) {
    console.error("ML Service error:", error.message);
    
    if (error.code === 'ECONNREFUSED') {
      return res.status(503).json({ error: "ML service is unavailable. Please try again later." });
    }
    
    if (error.response) {
      return res.status(error.response.status).json(error.response.data);
    }
    
    res.status(500).json({ error: "Failed to analyze text" });
  }
};
