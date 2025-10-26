import React, { useState } from 'react';
import { TrendingUp, Zap, Clock, Target, CheckCircle, ArrowRight, AlertCircle } from 'lucide-react';
import './ImprovementRoadmap.css';

function ImprovementRoadmap({ roadmap, weaknesses, strengths, executiveSummary }) {
  const [selectedPhase, setSelectedPhase] = useState('quick_wins');

  if (!roadmap) return null;

  const phases = [
    {
      id: 'quick_wins',
      title: 'Quick Wins',
      subtitle: 'Implement in 1-7 days',
      icon: <Zap size={24} />,
      color: '#10b981',
      data: roadmap.quick_wins || []
    },
    {
      id: 'medium_term',
      title: 'Medium Term',
      subtitle: 'Implement in 2-4 weeks',
      icon: <Clock size={24} />,
      color: '#f59e0b',
      data: roadmap.medium_term || []
    },
    {
      id: 'long_term',
      title: 'Long Term',
      subtitle: 'Strategic changes',
      icon: <Target size={24} />,
      color: '#8b5cf6',
      data: roadmap.long_term || []
    }
  ];

  const currentPhase = phases.find(p => p.id === selectedPhase);

  return (
    <div className="improvement-roadmap">
      <div className="roadmap-header">
        <div className="header-content">
          <h2>
            <TrendingUp size={28} />
            Improvement Roadmap
          </h2>
          <p>Strategic action plan to optimize your advertisement performance</p>
        </div>
        
        {executiveSummary && (
          <div className="roi-badge">
            <div className="roi-label">Expected ROI Increase</div>
            <div className="roi-value">{executiveSummary.estimated_roi_multiplier || '2-3x'}</div>
          </div>
        )}
      </div>

      {/* Biggest Opportunity Callout */}
      {executiveSummary?.biggest_opportunity && (
        <div className="opportunity-callout">
          <div className="callout-icon">
            <AlertCircle size={24} />
          </div>
          <div className="callout-content">
            <div className="callout-label">Biggest Opportunity</div>
            <div className="callout-text">{executiveSummary.biggest_opportunity}</div>
          </div>
        </div>
      )}

      {/* Strengths & Weaknesses Overview */}
      <div className="overview-grid">
        <div className="overview-card strengths-card">
          <h3>
            <CheckCircle size={20} />
            Key Strengths
          </h3>
          <div className="strength-list">
            {strengths?.slice(0, 3).map((strength, idx) => (
              <div key={idx} className="strength-item">
                <span className="strength-bullet">✓</span>
                {strength}
              </div>
            ))}
          </div>
        </div>

        <div className="overview-card weaknesses-card">
          <h3>
            <AlertCircle size={20} />
            Critical Weaknesses
          </h3>
          <div className="weakness-list">
            {weaknesses?.slice(0, 3).map((weakness, idx) => (
              <div key={idx} className="weakness-item">
                <span className="weakness-bullet">!</span>
                {weakness}
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Phase Selector */}
      <div className="phase-selector">
        {phases.map((phase) => (
          <button
            key={phase.id}
            className={`phase-tab ${selectedPhase === phase.id ? 'active' : ''}`}
            onClick={() => setSelectedPhase(phase.id)}
            style={{
              '--phase-color': phase.color,
              borderColor: selectedPhase === phase.id ? phase.color : '#e2e8f0'
            }}
          >
            <div className="phase-icon" style={{ color: phase.color }}>
              {phase.icon}
            </div>
            <div className="phase-info">
              <div className="phase-title">{phase.title}</div>
              <div className="phase-subtitle">{phase.subtitle}</div>
            </div>
            <div className="phase-count">{phase.data.length}</div>
          </button>
        ))}
      </div>

      {/* Action Steps */}
      <div className="action-steps">
        <div className="steps-header">
          <h3>{currentPhase.title} Action Steps</h3>
          <div className="steps-count">{currentPhase.data.length} actions</div>
        </div>

        <div className="steps-list">
          {currentPhase.data.map((action, idx) => (
            <div key={idx} className="action-card" style={{ '--delay': `${idx * 0.1}s` }}>
              <div className="action-number" style={{ background: currentPhase.color }}>
                {idx + 1}
              </div>
              
              <div className="action-content">
                <div className="action-header-row">
                  <h4 className="action-title">{action.action}</h4>
                  <div className="action-badges">
                    <span className={`priority-badge priority-${action.priority}`}>
                      Priority {action.priority}
                    </span>
                    <span className={`effort-badge effort-${action.effort}`}>
                      {action.effort} effort
                    </span>
                  </div>
                </div>
                
                <div className="action-impact">
                  <span className="impact-label">Expected Impact:</span>
                  <span className="impact-value">{action.impact}</span>
                </div>

                <div className="action-progress">
                  <div className="progress-bar">
                    <div 
                      className="progress-fill"
                      style={{ 
                        width: `${getImpactWidth(action.impact)}%`,
                        background: currentPhase.color
                      }}
                    />
                  </div>
                </div>
              </div>

              <button className="implement-btn" style={{ borderColor: currentPhase.color, color: currentPhase.color }}>
                <ArrowRight size={16} />
              </button>
            </div>
          ))}
        </div>
      </div>

      {/* Implementation Timeline */}
      <div className="timeline-section">
        <h3>Implementation Timeline</h3>
        <div className="timeline">
          {phases.map((phase, idx) => (
            <div key={phase.id} className="timeline-phase">
              <div className="timeline-dot" style={{ background: phase.color }} />
              <div className="timeline-content">
                <div className="timeline-phase-title">{phase.title}</div>
                <div className="timeline-actions">{phase.data.length} actions</div>
              </div>
              {idx < phases.length - 1 && (
                <div className="timeline-connector" />
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// Helper function to calculate impact bar width
function getImpactWidth(impact) {
  const impactText = impact.toLowerCase();
  
  if (impactText.includes('10%') || impactText.includes('high')) return 80;
  if (impactText.includes('15%') || impactText.includes('20%')) return 90;
  if (impactText.includes('25%') || impactText.includes('30%')) return 95;
  if (impactText.includes('5%') || impactText.includes('low')) return 50;
  
  // Try to extract percentage
  const match = impactText.match(/(\d+)%/);
  if (match) {
    return Math.min(parseInt(match[1]) * 3, 100);
  }
  
  return 70; // Default
}

export default ImprovementRoadmap;