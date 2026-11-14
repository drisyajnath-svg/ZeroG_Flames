import './App.css'; // This links the CSS file!
import React, { useState } from 'react';

export default function SpaceflightDashboard() {
  const [query, setQuery] = useState('');
  const [environment, setEnvironment] = useState('Moon (1/6g)');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSearch = async (e) => {
    e.preventDefault();
    setLoading(true);
    setResult(null);

    try {
      const response = await fetch('http://127.0.0.1:8000/ask_dashboard', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: query, environment: environment }),
      });

      const jsonResponse = await response.json();

      if (jsonResponse.status === 'success') {
        setResult(jsonResponse.data);
      }
    } catch (error) {
      console.error("Failed to fetch from API:", error);
      alert("Make sure the FastAPI backend is running on port 8000!");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="dashboard-container">
      <h1 className="dashboard-title">NASA Spaceflight Fire Safety Dashboard</h1>

      <form className="search-form" onSubmit={handleSearch}>
        <select
          className="environment-select"
          value={environment}
          onChange={(e) => setEnvironment(e.target.value)}
        >
          <option value="ISS (0g)">Microgravity - ISS (0g)</option>
          <option value="Moon (1/6g)">Lunar Base (1/6g)</option>
          <option value="Mars (3/8g)">Martian Base (3/8g)</option>
        </select>

        <input
          className="search-input"
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="e.g., How does 30% oxygen affect solid fuels?"
          required
        />

        <button className="search-button" type="submit">
          {loading ? 'Analyzing...' : 'Analyze'}
        </button>
      </form>

      {result && (
        <div className="results-card">
          <h2>AI Analysis Summary</h2>
          <p><strong>Verdict:</strong> {result.summary}</p>

          <h3>Key Variables Identified</h3>
          <ul>
            <li><strong>Fuel Type:</strong> {result.key_variables.fuel_type}</li>
            <li><strong>Oxygen Concentration:</strong> {result.key_variables.oxygen_concentration}</li>
            <li><strong>Gravity Level:</strong> {result.key_variables.gravity_level}</li>
          </ul>

          <h3>Gravity & Earth Comparisons</h3>
          <p>{result.comparisons}</p>

          {result.terrestrial_applications && (
            <>
              <h3>Earth Applications</h3>
              <p>{result.terrestrial_applications}</p>
            </>
          )}

          {result.confidence_warning && (
            <p className="warning-text"><strong>Note:</strong> {result.confidence_warning}</p>
          )}
        </div>
      )}
    </div>
  );
}