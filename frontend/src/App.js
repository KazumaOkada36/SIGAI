import React, { useState, useCallback } from 'react';
import { Upload, FileImage, FileVideo, Sparkles, TrendingUp, Eye, MessageSquare } from 'lucide-react';
import './App.css';

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [dragActive, setDragActive] = useState(false);

  const handleDrag = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  }, []);

  const handleFileInput = (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  const handleFile = (selectedFile) => {
    setFile(selectedFile);
    setResult(null);
    setError(null);

    // Create preview
    const reader = new FileReader();
    reader.onloadend = () => {
      setPreview(reader.result);
    };
    reader.readAsDataURL(selectedFile);
  };

  const analyzeAd = async () => {
    if (!file) return;

    setAnalyzing(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:5000/api/analyze', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Analysis failed');
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setAnalyzing(false);
    }
  };

  const reset = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
  };

  const getGradeColor = (grade) => {
    const colors = {
      'A': '#10b981',
      'B': '#3b82f6',
      'C': '#f59e0b',
      'D': '#ef4444',
      'F': '#dc2626'
    };
    return colors[grade] || '#6b7280';
  };

  return (
    <div className="App">
      {/* Header */}
      <header className="header">
        <div className="header-content">
          <div className="logo">
            <Sparkles size={32} />
            <h1>Ad Intelligence</h1>
          </div>
          <p className="tagline">AI-Powered Advertisement Analysis</p>
        </div>
      </header>

      <main className="main-content">
        {!result ? (
          <div className="upload-section">
            <div
              className={`drop-zone ${dragActive ? 'drag-active' : ''}`}
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
            >
              {!preview ? (
                <>
                  <Upload size={48} className="upload-icon" />
                  <h2>Upload Your Advertisement</h2>
                  <p>Drag & drop or click to select</p>
                  <div className="supported-formats">
                    <span><FileImage size={16} /> Images: PNG, JPG, GIF</span>
                    <span><FileVideo size={16} /> Videos: MP4, MOV, AVI</span>
                  </div>
                  <input
                    type="file"
                    accept="image/*,video/*"
                    onChange={handleFileInput}
                    className="file-input"
                  />
                </>
              ) : (
                <div className="preview-container">
                  {file.type.startsWith('image/') ? (
                    <img src={preview} alt="Preview" className="preview-image" />
                  ) : (
                    <video src={preview} controls className="preview-video" />
                  )}
                  <div className="preview-actions">
                    <button onClick={reset} className="btn btn-secondary">
                      Change File
                    </button>
                    <button
                      onClick={analyzeAd}
                      disabled={analyzing}
                      className="btn btn-primary"
                    >
                      {analyzing ? (
                        <>
                          <div className="spinner" />
                          Analyzing...
                        </>
                      ) : (
                        <>
                          <Sparkles size={20} />
                          Analyze Ad
                        </>
                      )}
                    </button>
                  </div>
                </div>
              )}
            </div>

            {error && (
              <div className="error-message">
                ⚠️ {error}
              </div>
            )}
          </div>
        ) : (
          <div className="results-section">
            {/* Grade Badge */}
            <div
              className="grade-badge"
              style={{ backgroundColor: getGradeColor(result.summary.grade) }}
            >
              <div className="grade-letter">{result.summary.grade}</div>
              <div className="grade-score">{result.summary.overall_score}/10</div>
            </div>

            {/* Summary Card */}
            <div className="summary-card">
              <h2>{result.summary.headline}</h2>
              <p className="summary-description">{result.summary.description}</p>
              
              <div className="insights-grid">
                {result.summary.key_insights.map((insight, idx) => (
                  <div key={idx} className="insight-item">
                    {insight}
                  </div>
                ))}
              </div>
            </div>

            {/* Detailed Features */}
            <div className="features-grid">
              {result.type === 'image' ? (
                <>
                  {/* Engagement Predictors */}
                  {result.features.engagement_predictors && (
                    <div className="feature-card">
                      <div className="feature-header">
                        <TrendingUp size={24} />
                        <h3>Engagement Signals</h3>
                      </div>
                      <div className="metrics">
                        <MetricBar
                          label="Scroll-Stopping Power"
                          value={result.features.engagement_predictors.scroll_stopping_power}
                        />
                        <MetricBar
                          label="Curiosity Gap"
                          value={result.features.engagement_predictors.curiosity_gap}
                        />
                        <MetricBar
                          label="Memability"
                          value={result.features.engagement_predictors.memability}
                        />
                      </div>
                    </div>
                  )}

                  {/* Emotional Signals */}
                  {result.features.emotional_signals && (
                    <div className="feature-card">
                      <div className="feature-header">
                        <MessageSquare size={24} />
                        <h3>Emotional Impact</h3>
                      </div>
                      <div className="metrics">
                        <div className="metric-row">
                          <span>Primary Emotion</span>
                          <span className="metric-value">
                            {result.features.emotional_signals.primary_emotion}
                          </span>
                        </div>
                        <MetricBar
                          label="Emotional Intensity"
                          value={result.features.emotional_signals.emotional_intensity}
                        />
                        <div className="metric-row">
                          <span>Creates FOMO</span>
                          <span className="metric-value">
                            {result.features.emotional_signals.creates_fomo ? '✅ Yes' : '❌ No'}
                          </span>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Visual Composition */}
                  {result.features.visual_composition && (
                    <div className="feature-card">
                      <div className="feature-header">
                        <Eye size={24} />
                        <h3>Visual Design</h3>
                      </div>
                      <div className="metrics">
                        <div className="metric-row">
                          <span>Color Scheme</span>
                          <span className="metric-value">
                            {result.features.visual_composition.color_scheme}
                          </span>
                        </div>
                        <MetricBar
                          label="Contrast Level"
                          value={result.features.visual_composition.contrast_level}
                        />
                        <MetricBar
                          label="Professional Polish"
                          value={result.features.visual_composition.professional_polish}
                        />
                      </div>
                    </div>
                  )}
                </>
              ) : (
                <>
                  {/* Video Features */}
                  {result.features.video_level_features && (
                    <>
                      <div className="feature-card">
                        <div className="feature-header">
                          <TrendingUp size={24} />
                          <h3>Video Performance</h3>
                        </div>
                        <div className="metrics">
                          <MetricBar
                            label="First 3-Sec Hook"
                            value={result.features.video_level_features.overall_engagement.first_3_seconds_hook}
                          />
                          <MetricBar
                            label="Avg Attention Retention"
                            value={result.features.video_level_features.overall_engagement.average_attention_retention}
                          />
                          <MetricBar
                            label="Completion Rate"
                            value={result.features.video_level_features.predicted_performance.estimated_completion_rate}
                          />
                        </div>
                      </div>

                      <div className="feature-card">
                        <div className="feature-header">
                          <Eye size={24} />
                          <h3>Video Dynamics</h3>
                        </div>
                        <div className="metrics">
                          <MetricBar
                            label="Motion Intensity"
                            value={result.features.video_level_features.video_dynamics.average_motion_intensity}
                          />
                          <div className="metric-row">
                            <span>Narrative Arc</span>
                            <span className="metric-value">
                              {result.features.video_level_features.video_dynamics.narrative_arc}
                            </span>
                          </div>
                          <div className="metric-row">
                            <span>Has Clear CTA</span>
                            <span className="metric-value">
                              {result.features.video_level_features.overall_engagement.has_clear_cta ? '✅ Yes' : '❌ No'}
                            </span>
                          </div>
                        </div>
                      </div>
                    </>
                  )}
                </>
              )}
            </div>

            <button onClick={reset} className="btn btn-primary analyze-another">
              Analyze Another Ad
            </button>
          </div>
        )}
      </main>

      <footer className="footer">
        <p>Powered by GPT-4 Vision via Lava • Built for AppLovin Challenge</p>
      </footer>
    </div>
  );
}

function MetricBar({ label, value, max = 10 }) {
  const percentage = (value / max) * 100;
  const color = percentage >= 70 ? '#10b981' : percentage >= 50 ? '#f59e0b' : '#ef4444';

  return (
    <div className="metric-bar">
      <div className="metric-label">
        <span>{label}</span>
        <span className="metric-value">{value}/{max}</span>
      </div>
      <div className="progress-bar">
        <div
          className="progress-fill"
          style={{ width: `${percentage}%`, backgroundColor: color }}
        />
      </div>
    </div>
  );
}

export default App;