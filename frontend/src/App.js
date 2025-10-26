import React, { useState } from 'react';
import LandingPage from './LandingPage';
import LoginPage from './LoginPage';
import SignupPage from './SignupPage';
import Dashboard from './Dashboard';
import HistoryPage from './HistoryPage';
import './App.css';

function App() {
  const [currentPage, setCurrentPage] = useState('landing'); // 'landing', 'login', 'signup', 'dashboard', 'history'
  const [user, setUser] = useState(null);
  const [selectedAnalysis, setSelectedAnalysis] = useState(null);

  const handleLogin = (userData) => {
    setUser(userData);
    setCurrentPage('dashboard');
  };

  const handleSignup = (userData) => {
    setUser(userData);
    setCurrentPage('dashboard');
  };

  const handleLogout = () => {
    setUser(null);
    setCurrentPage('landing');
  };

  const navigateTo = (page) => {
    setCurrentPage(page);
  };

  const viewAnalysis = (analysis) => {
    setSelectedAnalysis(analysis);
    setCurrentPage('view-analysis');
  };

  return (
    <div className="App">
      {currentPage === 'landing' && (
        <LandingPage onNavigate={navigateTo} />
      )}
      
      {currentPage === 'login' && (
        <LoginPage onNavigate={navigateTo} onLogin={handleLogin} />
      )}
      
      {currentPage === 'signup' && (
        <SignupPage onNavigate={navigateTo} onSignup={handleSignup} />
      )}
      
      {currentPage === 'dashboard' && (
        <Dashboard user={user} onLogout={handleLogout} onNavigate={navigateTo} />
      )}

      {currentPage === 'history' && (
        <HistoryPage user={user} onViewAnalysis={viewAnalysis} onNavigate={navigateTo} onLogout={handleLogout} />
      )}
    </div>
  );
}

export default App;