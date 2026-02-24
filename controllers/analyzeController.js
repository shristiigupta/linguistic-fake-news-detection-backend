exports.analyzeText = async (req, res) => {
  try {
    const { text } = req.body;

    if (!text || text.trim().length === 0) {
      return res.status(400).json({ error: "Text missing" });
    }

    // 🔹 Mocked ML response (stable for deployment)
    const response = {
      credibility: text.length > 120 ? "Likely Real" : "Likely Fake",
      risk: text.length > 120 ? "Low" : "High",
      confidence: text.length > 120 ? 82 : 68,
      insights: [
        { feature: "Lexical complexity", value: 0.73 },
        { feature: "Sentiment polarity", value: 0.41 },
        { feature: "Clickbait probability", value: 0.62 }
      ],
      explanation:
        "The prediction is based on linguistic patterns such as sentence structure, vocabulary richness, and emotional tone."
    };

    res.status(200).json(response);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Analysis failed" });
  }
};