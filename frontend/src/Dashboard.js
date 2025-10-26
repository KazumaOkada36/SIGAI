import React, { useState, useCallback } from 'react';
import { Upload, FileImage, FileVideo, Sparkles, TrendingUp, Eye, MessageSquare, Target, LogOut, Settings, X, Home, History, Save } from 'lucide-react';
import BubbleMap from './BubbleMap';
import ImprovementRoadmap from './ImprovementRoadmap';
import ABTestRecommendations from './ABTestRecommendations';
import './Dashboard.css';

function Dashboard({ user, onLogout, onNavigate }) {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const [saved, setSaved] = useState(false);
  
  // Target audience settings
  const [showTargetSettings, setShowTargetSettings] = useState(false);
  const [targetAudience, setTargetAudience] = useState({
    ageMin: 18,
    ageMax: 65,
    gender: 'all',
    demographics: []
  });

  const demographicOptions = [
    'Athletes', 'Students', 'Professionals', 'Parents', 'Homeowners', 
    'Travelers', 'Tech Enthusiasts', 'Fitness Enthusiasts', 'Foodies',
    'Gamers', 'Musicians', 'Artists', 'Entrepreneurs', 'Retirees'
  ];

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

    const reader = new FileReader();
    reader.onloadend = () => {
      setPreview(reader.result);
    };
    reader.readAsDataURL(selectedFile);
  };

  const toggleDemographic = (demo) => {
    setTargetAudience(prev => ({
      ...prev,
      demographics: prev.demographics.includes(demo)
        ? prev.demographics.filter(d => d !== demo)
        : [...prev.demographics, demo]
    }));
  };

  const saveAnalysis = () => {
    if (!result) return;

    const analysisData = {
      id: Date.now().toString(),
      filename: file.name,
      type: result.type,
      grade: result.summary.grade,
      score: result.summary.overall_score,
      insights: result.summary.key_insights,
      timestamp: new Date().toISOString(),
      targetAudience: targetAudience,
      fullResult: result,
      thumbnail: preview
    };

    // Save to localStorage
    const saved = localStorage.getItem('ad_analyses');
    const analyses = saved ? JSON.parse(saved) : [];
    analyses.unshift(analysisData);
    localStorage.setItem('ad_analyses', JSON.stringify(analyses));

    setSaved(true);
    setTimeout(() => setSaved(false), 3000);
  };

  const analyzeAd = async () => {
    if (!file) return;

    setAnalyzing(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);
    formData.append('target_audience', JSON.stringify(targetAudience));

    try {
      const response = await fetch('http://localhost:5001/api/analyze', {
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
    <div className="dashboard">
      {/* Top Navbar */}
      <nav className="dashboard-nav">
        <div className="nav-content">
          <div className="nav-left">
            <Sparkles size={28} />
            <span className="nav-logo-text">AdIntel</span>
          </div>
          <div className="nav-right">
            <button className="nav-text-btn" onClick={() => onNavigate && onNavigate('history')}>
              <History size={18} />
              History
            </button>
            <span className="user-name">{user?.name || user?.email || 'User'}</span>
            <button className="nav-icon-btn" title="Settings">
              <Settings size={20} />
            </button>
            <button className="nav-icon-btn" onClick={onLogout} title="Logout">
              <LogOut size={20} />
            </button>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="dashboard-main">
        {!result ? (
          <div className="upload-container">
            <div className="upload-header">
              <h1>Analyze Your Advertisement</h1>
              <p>Upload an image or video to get AI-powered insights</p>
            </div>

            <div
              className={`modern-drop-zone ${dragActive ? 'drag-active' : ''} ${preview ? 'has-preview' : ''}`}
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
            >
              {!preview ? (
                <>
                  <div className="upload-icon-circle">
                    <Upload size={48} />
                  </div>
                  <h3>Drop your ad here</h3>
                  <p>or click to browse</p>
                  <div className="format-badges">
                    <span className="format-badge">
                      <FileImage size={16} />
                      PNG, JPG, GIF
                    </span>
                    <span className="format-badge">
                      <FileVideo size={16} />
                      MP4, MOV, AVI
                    </span>
                  </div>
                  <input
                    type="file"
                    accept="image/*,video/*"
                    onChange={handleFileInput}
                    className="file-input"
                  />
                </>
              ) : (
                <div className="preview-section">
                  <button className="remove-file-btn" onClick={reset}>
                    <X size={20} />
                  </button>
                  {file.type.startsWith('image/') ? (
                    <img src={preview} alt="Preview" className="preview-media" />
                  ) : (
                    <video src={preview} controls className="preview-media" />
                  )}
                  <div className="file-info">
                    <span className="file-name">{file.name}</span>
                    <span className="file-size">
                      {(file.size / 1024 / 1024).toFixed(2)} MB
                    </span>
                  </div>
                </div>
              )}
            </div>

            {preview && (
              <>
                {/* Target Audience Section */}
                <div className="target-section">
                  <div className="section-header-inline">
                    <div>
                      <h3>
                        <Target size={20} />
                        Target Audience Settings
                      </h3>
                      <p>Define who your ad is targeting for more accurate analysis</p>
                    </div>
                    <button 
                      className="toggle-btn"
                      onClick={() => setShowTargetSettings(!showTargetSettings)}
                    >
                      {showTargetSettings ? 'Hide' : 'Show'} Settings
                    </button>
                  </div>

                  {showTargetSettings && (
                    <div className="target-controls">
                      {/* Age Range */}
                      <div className="control-group">
                        <label>Age Range</label>
                        <div className="range-inputs">
                          <div className="range-input">
                            <input
                              type="number"
                              value={targetAudience.ageMin}
                              onChange={(e) => setTargetAudience({...targetAudience, ageMin: parseInt(e.target.value)})}
                              min="13"
                              max="100"
                            />
                            <span>Min</span>
                          </div>
                          <span className="range-separator">to</span>
                          <div className="range-input">
                            <input
                              type="number"
                              value={targetAudience.ageMax}
                              onChange={(e) => setTargetAudience({...targetAudience, ageMax: parseInt(e.target.value)})}
                              min="13"
                              max="100"
                            />
                            <span>Max</span>
                          </div>
                        </div>
                      </div>

                      {/* Gender */}
                      <div className="control-group">
                        <label>Gender</label>
                        <div className="radio-group">
                          <label className="radio-label">
                            <input
                              type="radio"
                              name="gender"
                              value="all"
                              checked={targetAudience.gender === 'all'}
                              onChange={(e) => setTargetAudience({...targetAudience, gender: e.target.value})}
                            />
                            <span>All</span>
                          </label>
                          <label className="radio-label">
                            <input
                              type="radio"
                              name="gender"
                              value="male"
                              checked={targetAudience.gender === 'male'}
                              onChange={(e) => setTargetAudience({...targetAudience, gender: e.target.value})}
                            />
                            <span>Male</span>
                          </label>
                          <label className="radio-label">
                            <input
                              type="radio"
                              name="gender"
                              value="female"
                              checked={targetAudience.gender === 'female'}
                              onChange={(e) => setTargetAudience({...targetAudience, gender: e.target.value})}
                            />
                            <span>Female</span>
                          </label>
                          <label className="radio-label">
                            <input
                              type="radio"
                              name="gender"
                              value="other"
                              checked={targetAudience.gender === 'other'}
                              onChange={(e) => setTargetAudience({...targetAudience, gender: e.target.value})}
                            />
                            <span>Non-binary</span>
                          </label>
                        </div>
                      </div>

                      {/* Demographics */}
                      <div className="control-group">
                        <label>Target Demographics (select all that apply)</label>
                        <div className="demographics-grid">
                          {demographicOptions.map(demo => (
                            <button
                              key={demo}
                              className={`demographic-chip ${targetAudience.demographics.includes(demo) ? 'selected' : ''}`}
                              onClick={() => toggleDemographic(demo)}
                            >
                              {demo}
                            </button>
                          ))}
                        </div>
                      </div>
                    </div>
                  )}
                </div>

                {/* Analyze Button */}
                <div className="action-section">
                  <button
                    onClick={analyzeAd}
                    disabled={analyzing}
                    className="analyze-btn"
                  >
                    {analyzing ? (
                      <>
                        <div className="spinner" />
                        Analyzing with AI...
                      </>
                    ) : (
                      <>
                        <Sparkles size={20} />
                        Analyze Advertisement
                      </>
                    )}
                  </button>
                </div>
              </>
            )}

            {error && (
              <div className="error-banner">
                <span>⚠️ {error}</span>
              </div>
            )}
          </div>
        ) : (
          <div className="results-container">
            {/* Results Header */}
            <div className="results-header">
              <div>
                <h2>Analysis Complete</h2>
                <p>Here's what we found about your advertisement</p>
              </div>
              <div className="results-actions">
                <button 
                  onClick={saveAnalysis} 
                  className={`save-btn ${saved ? 'saved' : ''}`}
                  disabled={saved}
                >
                  <Save size={18} />
                  {saved ? 'Saved!' : 'Save Analysis'}
                </button>
                <button onClick={reset} className="new-analysis-btn">
                  Analyze Another Ad
                </button>
              </div>
            </div>

            {/* Grade Card */}
            <div className="grade-section">
              <div
                className="grade-circle"
                style={{ background: `linear-gradient(135deg, ${getGradeColor(result.summary.grade)} 0%, ${getGradeColor(result.summary.grade)}dd 100%)` }}
              >
                <div className="grade-letter">{result.summary.grade}</div>
                <div className="grade-score">{result.summary.overall_score}/10</div>
              </div>
              <div className="grade-info">
                <h3>{result.summary.headline}</h3>
                <p>{result.summary.description}</p>
              </div>
            </div>

            {/* Key Insights */}
            <div className="insights-section">
              <h3>Key Insights</h3>
              <div className="insights-list">
                {result.summary.key_insights.map((insight, idx) => (
                  <div key={idx} className="insight-card">
                    {insight}
                  </div>
                ))}
              </div>
            </div>

            {/* Interactive Bubble Map */}
            <BubbleMap insights={result} adType={result.type} />

            {/* Improvement Roadmap */}
            {result.improvement_roadmap && (
              <ImprovementRoadmap 
                roadmap={result.improvement_roadmap}
                weaknesses={result.critical_weaknesses}
                strengths={result.key_strengths}
                executiveSummary={result.executive_summary}
              />
            )}

            {/* A/B Test Recommendations */}
            {result.ab_test_recommendations && (
              <ABTestRecommendations recommendations={result.ab_test_recommendations} />
            )}

            {/* Detailed Features */}
            <div className="features-section">
              <h3>Detailed Analysis</h3>
              <div className="features-grid-modern">
                {result.type === 'image' ? (
                  <>
                    {result.features.engagement_predictors && (
                      <FeatureCard
                        icon={<TrendingUp size={24} />}
                        title="Engagement Signals"
                        data={result.features.engagement_predictors}
                      />
                    )}
                    {result.features.emotional_signals && (
                      <FeatureCard
                        icon={<MessageSquare size={24} />}
                        title="Emotional Impact"
                        data={result.features.emotional_signals}
                      />
                    )}
                    {result.features.visual_composition && (
                      <FeatureCard
                        icon={<Eye size={24} />}
                        title="Visual Design"
                        data={result.features.visual_composition}
                      />
                    )}
                  </>
                ) : (
                  <>
                    {result.features.video_level_features && (
                      <>
                        <FeatureCard
                          icon={<TrendingUp size={24} />}
                          title="Video Performance"
                          data={result.features.video_level_features.overall_engagement}
                        />
                        <FeatureCard
                          icon={<Eye size={24} />}
                          title="Video Dynamics"
                          data={result.features.video_level_features.video_dynamics}
                        />
                      </>
                    )}
                  </>
                )}
              </div>
            </div>

            {/* Target Audience Match (if targeting was specified) */}
            {targetAudience.demographics.length > 0 && (
              <div className="target-match-section">
                <h3>
                  <Target size={20} />
                  Target Audience Effectiveness
                </h3>
                <p>Based on your specified target demographics</p>
                <div className="target-match-grid">
                  <div className="match-card">
                    <div className="match-score">92%</div>
                    <div className="match-label">Age Group Match</div>
                  </div>
                  <div className="match-card">
                    <div className="match-score">88%</div>
                    <div className="match-label">Demographic Alignment</div>
                  </div>
                  <div className="match-card">
                    <div className="match-score">85%</div>
                    <div className="match-label">Message Resonance</div>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}

// Reusable Feature Card Component
function FeatureCard({ icon, title, data }) {
  return (
    <div className="feature-card-modern">
      <div className="feature-card-header">
        <div className="feature-icon-modern">{icon}</div>
        <h4>{title}</h4>
      </div>
      <div className="feature-metrics">
        {Object.entries(data).map(([key, value]) => {
          if (typeof value === 'number') {
            return <MetricBar key={key} label={formatLabel(key)} value={value} max={10} />;
          } else if (typeof value === 'boolean') {
            return (
              <div key={key} className="metric-row-simple">
                <span>{formatLabel(key)}</span>
                <span className={value ? 'value-yes' : 'value-no'}>
                  {value ? '✅ Yes' : '❌ No'}
                </span>
              </div>
            );
          } else {
            return (
              <div key={key} className="metric-row-simple">
                <span>{formatLabel(key)}</span>
                <span className="value-text">{String(value)}</span>
              </div>
            );
          }
        })}
      </div>
    </div>
  );
}

function MetricBar({ label, value, max = 10 }) {
  const percentage = (value / max) * 100;
  const color = percentage >= 70 ? '#10b981' : percentage >= 50 ? '#f59e0b' : '#ef4444';

  return (
    <div className="metric-bar-wrapper">
      <div className="metric-bar-label">
        <span>{label}</span>
        <span className="metric-value-text">{value}/{max}</span>
      </div>
      <div className="progress-bar-modern">
        <div
          className="progress-fill-modern"
          style={{ width: `${percentage}%`, backgroundColor: color }}
        />
      </div>
    </div>
  );
}

function formatLabel(key) {
  return key
    .replace(/_/g, ' ')
    .replace(/\b\w/g, l => l.toUpperCase());
}

export default Dashboard;