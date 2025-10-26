import React, { useState, useEffect } from 'react';
import { Calendar, FileImage, FileVideo, TrendingUp, Trash2, Eye, Download, Sparkles, LogOut, Settings, BarChart3 } from 'lucide-react';
import './History.css';

function HistoryPage({ user, onViewAnalysis, onNavigate, onLogout }) {
  const [analyses, setAnalyses] = useState([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState('all'); // 'all', 'image', 'video'
  const [sortBy, setSortBy] = useState('recent'); // 'recent', 'oldest', 'grade'

  useEffect(() => {
    loadHistory();
  }, []);

  const loadHistory = async () => {
    setLoading(true);
    try {
      // Load from backend
      const response = await fetch('http://localhost:5001/api/results');
      if (response.ok) {
        const data = await response.json();
        setAnalyses(data);
      }
    } catch (error) {
      console.error('Error loading history:', error);
      // Load from localStorage as fallback
      const saved = localStorage.getItem('ad_analyses');
      if (saved) {
        setAnalyses(JSON.parse(saved));
      }
    } finally {
      setLoading(false);
    }
  };

  const deleteAnalysis = (id) => {
    if (!window.confirm('Delete this analysis?')) return;
    
    const updated = analyses.filter(a => a.id !== id);
    setAnalyses(updated);
    localStorage.setItem('ad_analyses', JSON.stringify(updated));
  };

  const exportAnalysis = (analysis) => {
    const dataStr = JSON.stringify(analysis, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `analysis-${analysis.id}.json`;
    link.click();
  };

  const getFilteredAndSorted = () => {
    let filtered = analyses;

    // Filter by type
    if (filter !== 'all') {
      filtered = filtered.filter(a => a.type === filter);
    }

    // Sort
    filtered = [...filtered].sort((a, b) => {
      if (sortBy === 'recent') {
        return new Date(b.timestamp) - new Date(a.timestamp);
      } else if (sortBy === 'oldest') {
        return new Date(a.timestamp) - new Date(b.timestamp);
      } else if (sortBy === 'grade') {
        const gradeOrder = { 'A': 5, 'B': 4, 'C': 3, 'D': 2, 'F': 1 };
        return (gradeOrder[b.grade] || 0) - (gradeOrder[a.grade] || 0);
      }
      return 0;
    });

    return filtered;
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

  const formatDate = (timestamp) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diff = now - date;
    const days = Math.floor(diff / (1000 * 60 * 60 * 24));

    if (days === 0) return 'Today';
    if (days === 1) return 'Yesterday';
    if (days < 7) return `${days} days ago`;
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
  };

  const filteredAnalyses = getFilteredAndSorted();

  return (
    <div className="history-page">
      {/* Top Navbar */}
      <nav className="dashboard-nav">
        <div className="nav-content">
          <div className="nav-left">
            <Sparkles size={28} />
            <span className="nav-logo-text">AdIntel</span>
          </div>
          <div className="nav-right">
            <button className="nav-text-btn" onClick={() => onNavigate && onNavigate('dashboard')}>
              <BarChart3 size={18} />
              Analyze
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

      <div className="history-content">
      <div className="history-header">
        <div>
          <h1>Analysis History</h1>
          <p>View and manage your previous ad analyses</p>
        </div>
        <div className="history-stats">
          <div className="stat-box">
            <div className="stat-number">{analyses.length}</div>
            <div className="stat-label">Total Analyses</div>
          </div>
          <div className="stat-box">
            <div className="stat-number">
              {analyses.filter(a => a.grade === 'A' || a.grade === 'B').length}
            </div>
            <div className="stat-label">High Quality</div>
          </div>
        </div>
      </div>

      <div className="history-controls">
        <div className="filter-tabs">
          <button
            className={`filter-tab ${filter === 'all' ? 'active' : ''}`}
            onClick={() => setFilter('all')}
          >
            All ({analyses.length})
          </button>
          <button
            className={`filter-tab ${filter === 'image' ? 'active' : ''}`}
            onClick={() => setFilter('image')}
          >
            <FileImage size={16} />
            Images ({analyses.filter(a => a.type === 'image').length})
          </button>
          <button
            className={`filter-tab ${filter === 'video' ? 'active' : ''}`}
            onClick={() => setFilter('video')}
          >
            <FileVideo size={16} />
            Videos ({analyses.filter(a => a.type === 'video').length})
          </button>
        </div>

        <div className="sort-dropdown">
          <label>Sort by:</label>
          <select value={sortBy} onChange={(e) => setSortBy(e.target.value)}>
            <option value="recent">Most Recent</option>
            <option value="oldest">Oldest First</option>
            <option value="grade">Highest Grade</option>
          </select>
        </div>
      </div>

      {loading ? (
        <div className="loading-state">
          <div className="spinner-large"></div>
          <p>Loading your analyses...</p>
        </div>
      ) : filteredAnalyses.length === 0 ? (
        <div className="empty-state">
          <div className="empty-icon">📊</div>
          <h3>No analyses yet</h3>
          <p>Start by analyzing your first advertisement</p>
        </div>
      ) : (
        <div className="history-grid">
          {filteredAnalyses.map((analysis) => (
            <div key={analysis.id} className="history-card">
              <div className="card-image-section">
                {analysis.thumbnail ? (
                  <img src={analysis.thumbnail} alt={analysis.filename} className="card-thumbnail" />
                ) : (
                  <div className="card-placeholder">
                    {analysis.type === 'image' ? <FileImage size={48} /> : <FileVideo size={48} />}
                  </div>
                )}
                <div
                  className="card-grade-badge"
                  style={{ background: getGradeColor(analysis.grade) }}
                >
                  {analysis.grade}
                </div>
              </div>

              <div className="card-content">
                <div className="card-header">
                  <h3 className="card-title">{analysis.filename || 'Untitled'}</h3>
                  <div className="card-type">
                    {analysis.type === 'image' ? <FileImage size={14} /> : <FileVideo size={14} />}
                    {analysis.type}
                  </div>
                </div>

                <div className="card-meta">
                  <span className="meta-item">
                    <Calendar size={14} />
                    {formatDate(analysis.timestamp)}
                  </span>
                  <span className="meta-item">
                    <TrendingUp size={14} />
                    Score: {analysis.score}/10
                  </span>
                </div>

                <div className="card-insights">
                  {analysis.insights?.slice(0, 2).map((insight, idx) => (
                    <div key={idx} className="insight-preview">
                      {insight}
                    </div>
                  ))}
                </div>

                <div className="card-actions">
                  <button
                    className="action-btn primary"
                    onClick={() => onViewAnalysis(analysis)}
                  >
                    <Eye size={16} />
                    View Details
                  </button>
                  <button
                    className="action-btn secondary"
                    onClick={() => exportAnalysis(analysis)}
                  >
                    <Download size={16} />
                  </button>
                  <button
                    className="action-btn danger"
                    onClick={() => deleteAnalysis(analysis.id)}
                  >
                    <Trash2 size={16} />
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
      </div>
    </div>
  );
}

export default HistoryPage;