import React from 'react';
import { ArrowRight, Sparkles, Target, TrendingUp, Users, CheckCircle } from 'lucide-react';
import './LandingPage.css';

function LandingPage({ onNavigate }) {
  return (
    <div className="landing-page">
      {/* Navigation */}
      <nav className="navbar">
        <div className="nav-container">
          <div className="logo">
            <Sparkles size={28} />
            <span className="logo-text">AdIntel</span>
          </div>
          <div className="nav-links">
            <a href="#about">About</a>
            <a href="#how-it-works">How it Works</a>
            <a href="#testimonials">Testimonials</a>
            <button className="nav-btn-try" onClick={() => onNavigate('dashboard')}>
              Try for Free
            </button>
            <button className="nav-btn-login" onClick={() => onNavigate('login')}>
              Log In
            </button>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="hero-section">
        <div className="hero-container">
          <div className="hero-content">
            <div className="hero-badge">
              <Target size={16} />
              <span>AI-Powered Ad Intelligence</span>
            </div>
            <h1 className="hero-title">
              Optimize Your Ads with
              <span className="gradient-text"> AI Insights</span>
            </h1>
            <p className="hero-subtitle">
              Upload your advertisement and get instant AI-powered analysis on engagement, 
              targeting effectiveness, and performance predictions. Make data-driven decisions 
              to maximize your ROI.
            </p>
            <div className="hero-buttons">
              <button className="btn-primary-large" onClick={() => onNavigate('dashboard')}>
                Get Started Free
                <ArrowRight size={20} />
              </button>
              <button className="btn-secondary-large" onClick={() => {
                document.getElementById('how-it-works').scrollIntoView({ behavior: 'smooth' });
              }}>
                See How it Works
              </button>
            </div>
            <div className="hero-stats">
              <div className="stat-item">
                <div className="stat-number">10,000+</div>
                <div className="stat-label">Ads Analyzed</div>
              </div>
              <div className="stat-item">
                <div className="stat-number">95%</div>
                <div className="stat-label">Accuracy Rate</div>
              </div>
              <div className="stat-item">
                <div className="stat-number">2.5x</div>
                <div className="stat-label">Avg. Performance Boost</div>
              </div>
            </div>
          </div>
          <div className="hero-visual">
            <div className="floating-card card-1">
              <TrendingUp size={24} />
              <div>
                <div className="card-title">Engagement Score</div>
                <div className="card-value">8.7/10</div>
              </div>
            </div>
            <div className="floating-card card-2">
              <Target size={24} />
              <div>
                <div className="card-title">Target Match</div>
                <div className="card-value">92%</div>
              </div>
            </div>
            <div className="floating-card card-3">
              <Users size={24} />
              <div>
                <div className="card-title">Audience Fit</div>
                <div className="card-value">Excellent</div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* About Section */}
      <section id="about" className="section-about">
        <div className="section-container">
          <div className="section-header">
            <h2 className="section-title">Why AdIntel?</h2>
            <p className="section-subtitle">
              Transform your advertising strategy with AI-powered insights that go beyond basic metrics
            </p>
          </div>
          <div className="features-grid">
            <div className="feature-card">
              <div className="feature-icon">
                <Target />
              </div>
              <h3>Precision Targeting</h3>
              <p>Analyze how well your ad resonates with specific demographics, age groups, and audience segments</p>
            </div>
            <div className="feature-card">
              <div className="feature-icon">
                <TrendingUp />
              </div>
              <h3>Performance Prediction</h3>
              <p>Get AI-powered predictions on CTR, engagement rates, and conversion potential before launch</p>
            </div>
            <div className="feature-card">
              <div className="feature-icon">
                <Sparkles />
              </div>
              <h3>Creative Analysis</h3>
              <p>Deep insights into visual composition, emotional appeal, and scroll-stopping power</p>
            </div>
            <div className="feature-card">
              <div className="feature-icon">
                <Users />
              </div>
              <h3>Audience Matching</h3>
              <p>Understand which audience segments your ad appeals to most effectively</p>
            </div>
          </div>
        </div>
      </section>

      {/* How it Works */}
      <section id="how-it-works" className="section-how">
        <div className="section-container">
          <div className="section-header">
            <h2 className="section-title">How It Works</h2>
            <p className="section-subtitle">Get professional ad analysis in three simple steps</p>
          </div>
          <div className="steps-container">
            <div className="step-item">
              <div className="step-number">1</div>
              <div className="step-content">
                <h3>Upload Your Ad</h3>
                <p>Drag and drop your image or video advertisement. We support all major formats.</p>
              </div>
            </div>
            <div className="step-item">
              <div className="step-number">2</div>
              <div className="step-content">
                <h3>Define Your Target</h3>
                <p>Set your target audience parameters: age range, gender, demographics, and interests.</p>
              </div>
            </div>
            <div className="step-item">
              <div className="step-number">3</div>
              <div className="step-content">
                <h3>Get AI Insights</h3>
                <p>Receive comprehensive analysis with actionable recommendations to improve performance.</p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Testimonials */}
      <section id="testimonials" className="section-testimonials">
        <div className="section-container">
          <div className="section-header">
            <h2 className="section-title">Trusted by Marketing Teams</h2>
            <p className="section-subtitle">See what our users have to say</p>
          </div>
          <div className="testimonials-grid">
            <div className="testimonial-card">
              <div className="testimonial-stars">★★★★★</div>
              <p className="testimonial-text">
                "AdIntel helped us increase our CTR by 3x. The audience targeting insights were game-changing."
              </p>
              <div className="testimonial-author">
                <div className="author-info">
                  <div className="author-name">Sarah Chen</div>
                  <div className="author-title">Marketing Director, TechCorp</div>
                </div>
              </div>
            </div>
            <div className="testimonial-card">
              <div className="testimonial-stars">★★★★★</div>
              <p className="testimonial-text">
                "The AI predictions were incredibly accurate. We saved thousands by optimizing before launch."
              </p>
              <div className="testimonial-author">
                <div className="author-info">
                  <div className="author-name">Marcus Johnson</div>
                  <div className="author-title">Creative Lead, BrandHub</div>
                </div>
              </div>
            </div>
            <div className="testimonial-card">
              <div className="testimonial-stars">★★★★★</div>
              <p className="testimonial-text">
                "Finally, a tool that gives actionable insights instead of vanity metrics. Highly recommended!"
              </p>
              <div className="testimonial-author">
                <div className="author-info">
                  <div className="author-name">Emily Rodriguez</div>
                  <div className="author-title">Founder, GrowthLabs</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="section-cta">
        <div className="cta-container">
          <h2 className="cta-title">Ready to Optimize Your Ads?</h2>
          <p className="cta-subtitle">Join thousands of marketers using AI to improve their campaigns</p>
          <button className="btn-cta" onClick={() => onNavigate('dashboard')}>
            Start Analyzing Now
            <ArrowRight size={20} />
          </button>
        </div>
      </section>

      {/* Footer */}
      <footer className="footer">
        <div className="footer-container">
          <div className="footer-content">
            <div className="footer-brand">
              <div className="footer-logo">
                <Sparkles size={24} />
                <span>AdIntel</span>
              </div>
              <p>AI-Powered Advertisement Intelligence Platform</p>
            </div>
            <div className="footer-links">
              <div className="footer-column">
                <h4>Product</h4>
                <a href="#about">About</a>
                <a href="#how-it-works">How it Works</a>
                <a href="#testimonials">Testimonials</a>
              </div>
              <div className="footer-column">
                <h4>Company</h4>
                <a href="#about">About Us</a>
                <a href="#contact">Contact</a>
                <a href="#careers">Careers</a>
              </div>
              <div className="footer-column">
                <h4>Legal</h4>
                <a href="#privacy">Privacy Policy</a>
                <a href="#terms">Terms of Service</a>
              </div>
            </div>
          </div>
          <div className="footer-bottom">
            <p>© 2025 AdIntel. All rights reserved.</p>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default LandingPage;