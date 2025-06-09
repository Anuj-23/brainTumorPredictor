// src/App.js
import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [file, setFile] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [confidence, setConfidence] = useState(null);

  const handleFileChange = (e) => setFile(e.target.files[0]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      alert("Please select an MRI image.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
      setLoading(true);
      const response = await axios.post(
        `http://127.0.0.1:8000/predict`,
        formData,
        { headers: { 'Content-Type': 'multipart/form-data' } }
      );
      setPrediction(response.data.prediction);
      setConfidence(response.data.confidence);
    } catch (error) {
      console.error(error);
      alert("Error predicting image");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <h2 className='title'>ViT-based MRI Brain Tumor Classifier</h2>
      <form onSubmit={handleSubmit}>
        <input className="input" type="file" accept="image/*" onChange={handleFileChange} />
        <button className="button" type="submit" disabled={loading}>
          {loading ? "Predicting..." : "Predict"}
        </button>
      </form>

      {prediction && (
        <div style={{ marginTop: 20 }}>
          <h3>Prediction: {prediction}</h3>
          
          <h3>Confidence: {confidence}</h3>
          
        </div>
        
      )}
    </div>
  );
}

export default App;
