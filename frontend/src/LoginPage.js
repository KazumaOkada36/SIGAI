import React, { useState } from 'react';
import { Sparkles, Mail, Lock, ArrowLeft } from 'lucide-react';
import './Auth.css';

function LoginPage({ onNavigate, onLogin }) {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    // For now, just log in without validation (you can add backend auth later)
    if (email && password) {
      onLogin({ email });
      onNavigate('dashboard');
    }
  };

  return (
    <div className="auth-page">
      <div className="auth-container">
        <div className="auth-left">
          <div className="auth-brand">
            <Sparkles size={40} />
            <h1>AdIntel</h1>
          </div>
          <div className="auth-content">
            <h2>AI-Powered Advertisement Intelligence</h2>
            <p>Optimize your campaigns with data-driven insights</p>
            <div className="auth-features">
              <div className="auth-feature">✓ Instant AI Analysis</div>
              <div className="auth-feature">✓ Target Audience Insights</div>
              <div className="auth-feature">✓ Performance Predictions</div>
            </div>
          </div>
        </div>

        <div className="auth-right">
          <button className="back-button" onClick={() => onNavigate('landing')}>
            <ArrowLeft size={20} />
            Back
          </button>

          <div className="auth-form-container">
            <div className="auth-header">
              <h2>Welcome back!</h2>
              <p>Enter your details to access your account</p>
            </div>

            <form onSubmit={handleSubmit} className="auth-form">
              <div className="form-group">
                <label>Email Address</label>
                <div className="input-wrapper">
                  <Mail size={20} />
                  <input
                    type="email"
                    placeholder="Enter your email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    required
                  />
                </div>
              </div>

              <div className="form-group">
                <label>Password</label>
                <div className="input-wrapper">
                  <Lock size={20} />
                  <input
                    type="password"
                    placeholder="Enter your password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    required
                  />
                </div>
              </div>

              <div className="form-options">
                <label className="checkbox-label">
                  <input type="checkbox" />
                  <span>Remember me</span>
                </label>
                <a href="#forgot" className="forgot-link">Forgot password?</a>
              </div>

              <button type="submit" className="auth-submit-btn">
                Log In
              </button>
            </form>

            <div className="auth-divider">
              <span>or</span>
            </div>

            <button className="google-btn">
              <svg width="20" height="20" viewBox="0 0 24 24">
                <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
                <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
                <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/>
                <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
              </svg>
              Continue with Google
            </button>

            <div className="auth-footer">
              <p>
                Don't have an account? 
                <button onClick={() => onNavigate('signup')} className="link-btn">
                  Sign up
                </button>
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default LoginPage;