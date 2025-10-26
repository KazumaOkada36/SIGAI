import React from 'react';
import { FlaskConical, TrendingUp, Target, Lightbulb } from 'lucide-react';
import './ABTestRecommendations.css';

function ABTestRecommendations({ recommendations }) {
  if (!recommendations || recommendations.length === 0) return null;

  return (
    <div className="ab-test-recommendations">
      <div className="ab-header">
        <h2>
          <FlaskConical size={28} />
          A/B Test Recommendations
        </h2>
        <p>Scientifically validated opportunities to improve performance</p>
      </div>

      <div className="test-grid">
        {recommendations.map((test, idx) => (
          <div key={idx} className="test-card" style={{ '--delay': `${idx * 0.15}s` }}>
            <div className="test-number">Test #{idx + 1}</div>
            
            <div className="test-content">
              <h3 className="test-title">
                <Lightbulb size={20} />
                {test.test}
              </h3>

              <div className="test-section">
                <div className="section-label">
                  <Target size={16} />
                  Hypothesis
                </div>
                <p className="hypothesis-text">{test.hypothesis}</p>
              </div>

              <div className="test-section">
                <div className="section-label">
                  <TrendingUp size={16} />
                  Expected Lift
                </div>
                <div className="lift-badge">{test.expected_lift}</div>
              </div>

              <div className="variant-box">
                <div className="variant-header">Variant Suggestion</div>
                <div className="variant-content">{test.variant_suggestion}</div>
              </div>

              <div className="test-metrics">
                <div className="metric-item">
                  <span className="metric-label">Confidence</span>
                  <div className="confidence-bar">
                    <div 
                      className="confidence-fill"
                      style={{ width: `${getConfidenceLevel(test.expected_lift)}%` }}
                    />
                  </div>
                </div>
                <div className="metric-item">
                  <span className="metric-label">Effort</span>
                  <span className="effort-badge">Low</span>
                </div>
              </div>
            </div>

            <button className="run-test-btn">
              Set Up Test
            </button>
          </div>
        ))}
      </div>

      <div className="ab-footer">
        <div className="footer-tip">
          <strong>Pro Tip:</strong> Run these tests sequentially, starting with the highest expected lift.
          Wait for statistical significance (minimum 1000 impressions per variant) before concluding.
        </div>
      </div>
    </div>
  );
}

// Calculate confidence level from expected lift
function getConfidenceLevel(lift) {
  const liftValue = parseInt(lift) || 10;
  if (liftValue >= 20) return 90;
  if (liftValue >= 15) return 80;
  if (liftValue >= 10) return 70;
  if (liftValue >= 5) return 60;
  return 50;
}

export default ABTestRecommendations;