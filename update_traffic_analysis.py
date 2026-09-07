import re

file_path = "frontend/src/pages/TrafficAnalysis.tsx"
with open(file_path, "r") as f:
    content = f.read()

# 1. Add state for analysis result
state_addition = """  const [dragOver, setDragOver]             = useState(false);
  const [analysisResult, setAnalysisResult] = useState<any>(null);"""
content = content.replace("  const [dragOver, setDragOver]             = useState(false);", state_addition)

# 2. Capture analysis result in handleFile
handle_file_addition = """      const result = await uploadTrafficFile(file, (pct, stage) => {
        setUploadProgress(pct);
        setUploadStage(stage);
      });
      setUploadDone(true);
      setUploadMsg(result.message);
      if (result.data) {
        setAnalysisResult(result.data);
      }"""
content = content.replace("      const result = await uploadTrafficFile(file, (pct, stage) => {\n        setUploadProgress(pct);\n        setUploadStage(stage);\n      });\n      setUploadDone(true);\n      setUploadMsg(result.message);", handle_file_addition)

# 3. Add Feature Coverage UI panel below the upload section
ui_addition = """
      {analysisResult && (
        <div className="mb-4">
          <Card>
            <CardHeader title="Feature Compatibility & Model Selection" icon={<BarChart2 size={14} />} />
            <div className="p-4 space-y-4">
              <div className={`p-4 rounded-lg border ${analysisResult.model_mode === 'gru_60' ? 'border-safe-dim bg-safe-bg text-safe' : 'border-forecast-dim bg-forecast-bg text-forecast'}`}>
                <h3 className="font-semibold mb-2">
                  Model Mode: {analysisResult.model_mode === 'gru_60' ? '60-Feature GRU Forecast (Full Analysis)' : 'Partial-Feature PyTorch ML (Fallback)'}
                </h3>
                {analysisResult.model_mode !== 'gru_60' && (
                  <p className="text-sm opacity-90 mb-2">
                    ⚠️ The uploaded dataset lacked the required features for full GRU analysis. Prediction was made using a limited fallback model. 8-step temporal forecast is unavailable.
                  </p>
                )}
                <div className="grid grid-cols-2 gap-4 text-sm mt-3">
                  <div>
                    <span className="opacity-80 block text-xs">Required Features</span>
                    <span className="font-mono text-lg">{analysisResult.feature_coverage?.required || 60}</span>
                  </div>
                  <div>
                    <span className="opacity-80 block text-xs">Features Found (Raw)</span>
                    <span className="font-mono text-lg">{analysisResult.feature_coverage?.available || 0}</span>
                  </div>
                  <div>
                    <span className="opacity-80 block text-xs">Engineered Features</span>
                    <span className="font-mono text-lg">{analysisResult.feature_coverage?.engineered || 0}</span>
                  </div>
                  <div>
                    <span className="opacity-80 block text-xs">Total Valid Features</span>
                    <span className="font-mono text-lg">{analysisResult.feature_coverage?.final || 0}</span>
                  </div>
                </div>
              </div>
            </div>
          </Card>
        </div>
      )}

      {/* Charts row */}"""
content = content.replace("      {/* Charts row */}", ui_addition)

with open(file_path, "w") as f:
    f.write(content)
print("Updated TrafficAnalysis.tsx")
