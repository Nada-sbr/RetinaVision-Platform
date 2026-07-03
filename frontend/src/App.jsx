import React, { useState, useEffect, useRef } from 'react';
import { 
  Activity, User, Users, Upload, History, LogOut, CheckCircle, 
  AlertTriangle, ArrowRight, Eye, ShieldAlert, FileText, CheckSquare, Trash2
} from 'lucide-react';

// If running in development (Vite port 5173), target backend port 8000.
// In production (Nginx proxy on port 80), use relative paths to bypass CORS.
const BACKEND_URL = window.location.port === "5173" ? "http://localhost:8000" : "";

// Simple Markdown parser for clinical reports
function renderMarkdown(text) {
  if (!text) return "";
  
  // Replace headers
  let html = text
    .replace(/^# (.*$)/gim, '<h1>$1</h1>')
    .replace(/^## (.*$)/gim, '<h2>$1</h2>')
    .replace(/^### (.*$)/gim, '<h3>$1</h3>')
    .replace(/^\*\* (.*$)/gim, '<strong>$1</strong>');
    
  // Replace bold
  html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
  
  // Replace bullet points
  html = html.replace(/^\* (.*$)/gim, '<li>$1</li>');
  html = html.replace(/<\/li>\n<li>/g, '</li><li>'); // group list items
  
  // Wrap list items in ul
  html = html.replace(/(<li>.*<\/li>)/g, '<ul>$1</ul>');
  
  // Replace line breaks
  html = html.replace(/\n/g, '<br />');
  
  return <div dangerouslySetInnerHTML={{ __html: html }} />;
}

function App() {
  // Auth state
  const [token, setToken] = useState(localStorage.getItem('token') || null);
  const [currentUser, setCurrentUser] = useState(null);
  const [isRegisterMode, setIsRegisterMode] = useState(false);
  
  // Form states for login/register
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [fullName, setFullName] = useState('');
  const [authError, setAuthError] = useState('');
  
  // Navigation
  const [activeTab, setActiveTab] = useState('dashboard');
  
  // Data states
  const [patients, setPatients] = useState([]);
  const [history, setHistory] = useState([]);
  const [recentExams, setRecentExams] = useState([]);
  const [errorMsg, setErrorMsg] = useState('');
  const [successMsg, setSuccessMsg] = useState('');
  
  // Patient creation states
  const [patientFirstName, setPatientFirstName] = useState('');
  const [patientLastName, setPatientLastName] = useState('');
  const [patientBirthDate, setPatientBirthDate] = useState('');
  const [patientGender, setPatientGender] = useState('Male');
  
  // Upload states
  const [uploadPatientId, setUploadPatientId] = useState('');
  const [uploadEye, setUploadEye] = useState('left');
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  
  // Analysis states
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisStep, setAnalysisStep] = useState('');
  const [analysisResult, setAnalysisResult] = useState(null);
  
  const fileInputRef = useRef(null);

  // Load initial data if authenticated
  useEffect(() => {
    if (token) {
      // Validate user / fetch profile
      fetchUserProfile();
      fetchPatients();
      fetchHistory();
    }
  }, [token]);

  const fetchUserProfile = async () => {
    // For simplicity, we decode JWT subject or set mock doctor name
    setCurrentUser({ full_name: "Dr. Ophthalmo", role: "Ophthalmologist" });
  };

  const fetchPatients = async () => {
    try {
      const response = await fetch(`${BACKEND_URL}/api/history`, {
        headers: { "Authorization": `Bearer ${token}` }
      });
      if (response.ok) {
        const historyData = await response.json();
        // Extract unique patients from history
        const uniquePatients = [];
        const seen = new Set();
        historyData.forEach(item => {
          if (!seen.has(item.patient.id)) {
            seen.add(item.patient.id);
            uniquePatients.push(item.patient);
          }
        });
        setPatients(uniquePatients);
      }
    } catch (err) {
      console.error("Error fetching patients", err);
    }
  };

  const fetchHistory = async () => {
    try {
      const response = await fetch(`${BACKEND_URL}/api/history`, {
        headers: { "Authorization": `Bearer ${token}` }
      });
      if (response.ok) {
        const data = await response.json();
        setHistory(data);
        setRecentExams(data.slice(0, 3));
      }
    } catch (err) {
      console.error("Error fetching history", err);
    }
  };

  // Auth actions
  const handleLogin = async (e) => {
    e.preventDefault();
    setAuthError('');
    try {
      const formData = new URLSearchParams();
      formData.append('username', email);
      formData.append('password', password);
      
      const response = await fetch(`${BACKEND_URL}/api/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: formData
      });
      
      if (response.ok) {
        const data = await response.json();
        localStorage.setItem('token', data.access_token);
        setToken(data.access_token);
      } else {
        const err = await response.json();
        setAuthError(err.detail || "Authentication failed");
      }
    } catch (err) {
      setAuthError("Could not connect to backend server");
    }
  };

  const handleRegister = async (e) => {
    e.preventDefault();
    setAuthError('');
    try {
      const response = await fetch(`${BACKEND_URL}/api/register`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          email,
          password,
          full_name: fullName,
          role: "ophthalmologist"
        })
      });
      
      if (response.ok) {
        setIsRegisterMode(false);
        setSuccessMsg("Registration successful! Please log in.");
        setTimeout(() => setSuccessMsg(''), 5000);
      } else {
        const err = await response.json();
        setAuthError(err.detail || "Registration failed");
      }
    } catch (err) {
      setAuthError("Could not connect to backend server");
    }
  };

  const handleLogout = () => {
    localStorage.removeItem('token');
    setToken(null);
    setCurrentUser(null);
    setAnalysisResult(null);
  };

  // Patient Registration
  const handleRegisterPatient = async (e) => {
    e.preventDefault();
    setErrorMsg('');
    setSuccessMsg('');
    try {
      const response = await fetch(`${BACKEND_URL}/api/patients`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({
          first_name: patientFirstName,
          last_name: patientLastName,
          birth_date: patientBirthDate,
          gender: patientGender
        })
      });
      
      if (response.ok) {
        const newPatient = await response.json();
        setSuccessMsg(`Patient ${newPatient.first_name} ${newPatient.last_name} registered successfully!`);
        setPatients(prev => [newPatient, ...prev]);
        setPatientFirstName('');
        setPatientLastName('');
        setPatientBirthDate('');
        
        // Redirect to Upload
        setUploadPatientId(newPatient.id);
        setActiveTab('upload');
        setTimeout(() => setSuccessMsg(''), 5000);
      } else {
        const err = await response.json();
        setErrorMsg(err.detail || "Failed to register patient");
      }
    } catch (err) {
      setErrorMsg("Error communicating with backend");
    }
  };

  // File handling
  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
    }
  };

  // Drag and drop
  const handleDragOver = (e) => e.preventDefault();
  const handleDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
    }
  };

  // Deep Learning & LLM Execution pipeline
  const runOcularAnalysis = async () => {
    setErrorMsg('');
    if (!uploadPatientId) {
      setErrorMsg("Please select a patient.");
      return;
    }
    if (!selectedFile) {
      setErrorMsg("Please upload a fundus image.");
      return;
    }

    setIsAnalyzing(true);
    setAnalysisResult(null);
    
    try {
      // Step 1: Upload image file
      setAnalysisStep("1. Chargement de l'image couleur du fond d'œil...");
      const formData = new FormData();
      formData.append('patient_id', uploadPatientId);
      formData.append('eye', uploadEye);
      formData.append('file', selectedFile);

      const uploadRes = await fetch(`${BACKEND_URL}/api/upload-image`, {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${token}` },
        body: formData
      });

      if (!uploadRes.ok) {
        const err = await uploadRes.json();
        throw new Error(err.detail || "Failed to upload image");
      }
      const imageData = await uploadRes.json();
      const imageId = imageData.id;

      // Step 2: CNN Infiltration (ResNet-50 classification)
      setAnalysisStep("2. Inférence du modèle de vision (ResNet-50) et prédictions...");
      const predictRes = await fetch(`${BACKEND_URL}/api/predict/${imageId}`, {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${token}` }
      });

      if (!predictRes.ok) {
        const err = await predictRes.json();
        throw new Error(err.detail || "ResNet-50 prediction task failed.");
      }
      const predictionData = await predictRes.json();

      // Step 3: Explainable AI generation (Grad-CAM overlays)
      setAnalysisStep("3. Extraction des gradients et génération de l'explicabilité Grad-CAM...");
      const explainRes = await fetch(`${BACKEND_URL}/api/explain/${imageId}`, {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${token}` }
      });

      if (!explainRes.ok) {
        const err = await explainRes.json();
        throw new Error(err.detail || "Grad-CAM generation failed.");
      }
      const explainedImageData = await explainRes.json();

      // Step 4: Clinical Report compilation (LLM Mistral via Ollama)
      setAnalysisStep("4. Compilation clinique et génération du rapport médical par le LLM (Mistral)...");
      const reportRes = await fetch(`${BACKEND_URL}/api/report/${imageId}`, {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${token}` }
      });

      if (!reportRes.ok) {
        const err = await reportRes.json();
        throw new Error(err.detail || "Clinical report generation failed.");
      }
      const reportData = await reportRes.json();

      // Complete Result compiled
      setAnalysisResult({
        image: explainedImageData,
        prediction: predictionData,
        report: reportData,
        patient: patients.find(p => p.id === parseInt(uploadPatientId))
      });

      // Clear upload form
      setSelectedFile(null);
      setPreviewUrl(null);
      
      // Update tables
      fetchHistory();

    } catch (err) {
      setErrorMsg(err.message || "An error occurred during ocular pipeline analysis.");
    } finally {
      setIsAnalyzing(false);
      setAnalysisStep('');
    }
  };

  const viewHistoryExam = (item) => {
    setAnalysisResult({
      image: item.image,
      prediction: item.prediction,
      report: item.report,
      patient: item.patient
    });
    setActiveTab('upload');
  };

  // Render Functions
  if (!token) {
    return (
      <div className="auth-container">
        <div className="auth-card glass">
          <div className="auth-header">
            <span className="logo-icon"><Activity /></span>
            <h1>Ocular AI Portal</h1>
            <p>Plateforme de Diagnostic Ophtalmologique par IA</p>
          </div>
          
          {authError && <div className="alert alert-danger">{authError}</div>}
          {successMsg && <div className="alert alert-success">{successMsg}</div>}

          <form onSubmit={isRegisterMode ? handleRegister : handleLogin}>
            {isRegisterMode && (
              <div className="form-group">
                <label>Nom complet</label>
                <input 
                  type="text" 
                  className="form-control" 
                  value={fullName}
                  onChange={(e) => setFullName(e.target.value)}
                  placeholder="Dr. Jean Dupont"
                  required
                />
              </div>
            )}
            
            <div className="form-group">
              <label>Adresse Email</label>
              <input 
                type="email" 
                className="form-control" 
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="clinique@hopital.com"
                required
              />
            </div>
            
            <div className="form-group">
              <label>Mot de passe</label>
              <input 
                type="password" 
                className="form-control" 
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="••••••••"
                required
              />
            </div>

            <button type="submit" className="btn btn-primary">
              {isRegisterMode ? "Créer un compte" : "Se connecter"} <ArrowRight size={16} />
            </button>
          </form>

          <div className="auth-footer">
            {isRegisterMode ? (
              <p>Déjà inscrit ? <span onClick={() => setIsRegisterMode(false)}>Se connecter</span></p>
            ) : (
              <p>Pas de compte ? <span onClick={() => setIsRegisterMode(true)}>Créer un compte</span></p>
            )}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="dashboard-layout">
      {/* Sidebar Navigation */}
      <div className="sidebar glass">
        <div>
          <div className="sidebar-brand">
            <Activity className="logo-icon" style={{margin: 0}} />
            <h2>Ocular AI</h2>
          </div>
          
          <ul className="nav-menu">
            <li 
              className={`nav-item ${activeTab === 'dashboard' ? 'active' : ''}`}
              onClick={() => { setActiveTab('dashboard'); setAnalysisResult(null); }}
            >
              <Activity size={18} /> Tableau de Bord
            </li>
            <li 
              className={`nav-item ${activeTab === 'register_patient' ? 'active' : ''}`}
              onClick={() => { setActiveTab('register_patient'); setAnalysisResult(null); }}
            >
              <Users size={18} /> Enregistrer Patient
            </li>
            <li 
              className={`nav-item ${activeTab === 'upload' ? 'active' : ''}`}
              onClick={() => { setActiveTab('upload'); }}
            >
              <Upload size={18} /> Analyse Rétinienne
            </li>
            <li 
              className={`nav-item ${activeTab === 'history' ? 'active' : ''}`}
              onClick={() => { setActiveTab('history'); setAnalysisResult(null); }}
            >
              <History size={18} /> Historique Examens
            </li>
          </ul>
        </div>
        
        <div className="sidebar-footer">
          <div className="user-badge">
            <div className="avatar">
              {currentUser?.full_name?.charAt(4).toUpperCase()}
            </div>
            <div className="user-info">
              <p>{currentUser?.full_name}</p>
              <span>{currentUser?.role}</span>
            </div>
          </div>
          
          <button onClick={handleLogout} className="btn btn-secondary" style={{padding: '8px 16px', fontSize: '0.85rem'}}>
            <LogOut size={16} /> Déconnexion
          </button>
        </div>
      </div>

      {/* Main Panel Content */}
      <div className="main-content">
        
        {/* Error notification */}
        {errorMsg && <div className="alert alert-danger">{errorMsg}</div>}
        {successMsg && <div className="alert alert-success">{successMsg}</div>}

        {/* Tab 1: Dashboard overview */}
        {activeTab === 'dashboard' && (
          <div>
            <div className="content-header">
              <h1>Tableau de Bord Clinique</h1>
            </div>

            <div className="grid-3">
              <div className="card glass">
                <div className="card-title"><Users size={20} /> Patients Totaux</div>
                <h2 style={{fontSize: '2.5rem', fontWeight: 700}}>{patients.length}</h2>
                <p style={{color: 'var(--text-secondary)', fontSize: '0.85rem', marginTop: '10px'}}>Profils de patients enregistrés</p>
              </div>
              <div className="card glass">
                <div className="card-title"><Activity size={20} /> Analyses Réalisées</div>
                <h2 style={{fontSize: '2.5rem', fontWeight: 700}}>{history.length}</h2>
                <p style={{color: 'var(--text-secondary)', fontSize: '0.85rem', marginTop: '10px'}}>Examens d'imagerie rétinienne</p>
              </div>
              <div className="card glass">
                <div className="card-title"><FileText size={20} /> Rapports Générés</div>
                <h2 style={{fontSize: '2.5rem', fontWeight: 700}}>
                  {history.filter(h => h.report).length}
                </h2>
                <p style={{color: 'var(--text-secondary)', fontSize: '0.85rem', marginTop: '10px'}}>Comptes-rendus d'expertise LLM</p>
              </div>
            </div>

            <div className="card glass" style={{marginTop: '24px'}}>
              <div className="card-title"><History size={20} /> Examens Récents</div>
              
              {recentExams.length === 0 ? (
                <p style={{color: 'var(--text-secondary)', textAlign: 'center', padding: '20px'}}>Aucun examen rétinien réalisé pour le moment.</p>
              ) : (
                <table className="history-table">
                  <thead>
                    <tr>
                      <th>Date</th>
                      <th>Patient</th>
                      <th>Œil</th>
                      <th>Pathologie Cible (Confiance)</th>
                      <th>Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {recentExams.map((item, idx) => {
                      const probs = item.prediction ? [
                        { name: 'Normal', val: item.prediction.n_prob },
                        { name: 'Diabète', val: item.prediction.d_prob },
                        { name: 'Glaucome', val: item.prediction.g_prob },
                        { name: 'Cataracte', val: item.prediction.c_prob },
                        { name: 'DMLA', val: item.prediction.a_prob },
                        { name: 'Hypertension', val: item.prediction.h_prob },
                        { name: 'Myopie', val: item.prediction.m_prob },
                        { name: 'Autre', val: item.prediction.o_prob }
                      ] : [];
                      
                      const topClass = probs.length > 0 ? probs.sort((a,b) => b.val - a.val)[0] : null;

                      return (
                        <tr key={idx}>
                          <td>{new Date(item.image.uploaded_at).toLocaleDateString('fr-FR')}</td>
                          <td>{item.patient.first_name} {item.patient.last_name}</td>
                          <td style={{textTransform: 'capitalize'}}>{item.image.eye === 'left' ? 'Gauche' : 'Droit'}</td>
                          <td>
                            {topClass ? (
                              <span className={topClass.name === 'Normal' ? 'tag tag-normal' : 'tag tag-pathology'}>
                                {topClass.name} ({Math.round(topClass.val * 100)}%)
                              </span>
                            ) : 'Calcul en cours'}
                          </td>
                          <td>
                            <button 
                              onClick={() => viewHistoryExam(item)}
                              className="btn btn-secondary" 
                              style={{padding: '6px 12px', fontSize: '0.8rem', width: 'auto'}}
                            >
                              <Eye size={14} /> Consulter
                            </button>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              )}
            </div>
          </div>
        )}

        {/* Tab 2: Register Patient */}
        {activeTab === 'register_patient' && (
          <div style={{maxWidth: '600px', margin: '0 auto'}}>
            <div className="content-header">
              <h1>Enregistrer un Nouveau Patient</h1>
            </div>

            <div className="card glass">
              <form onSubmit={handleRegisterPatient}>
                <div className="grid-2">
                  <div className="form-group">
                    <label>Prénom</label>
                    <input 
                      type="text" 
                      className="form-control"
                      value={patientFirstName}
                      onChange={(e) => setPatientFirstName(e.target.value)}
                      placeholder="Jean"
                      required
                    />
                  </div>
                  <div className="form-group">
                    <label>Nom</label>
                    <input 
                      type="text" 
                      className="form-control"
                      value={patientLastName}
                      onChange={(e) => setPatientLastName(e.target.value)}
                      placeholder="Dupont"
                      required
                    />
                  </div>
                </div>

                <div className="form-group">
                  <label>Date de Naissance</label>
                  <input 
                    type="date" 
                    className="form-control"
                    value={patientBirthDate}
                    onChange={(e) => setPatientBirthDate(e.target.value)}
                    required
                  />
                </div>

                <div className="form-group">
                  <label>Sexe</label>
                  <select 
                    className="form-control"
                    value={patientGender}
                    onChange={(e) => setPatientGender(e.target.value)}
                    required
                  >
                    <option value="Male">Masculin</option>
                    <option value="Female">Féminin</option>
                    <option value="Other">Autre</option>
                  </select>
                </div>

                <button type="submit" className="btn btn-primary" style={{marginTop: '10px'}}>
                  <CheckSquare size={16} /> Enregistrer la fiche patient
                </button>
              </form>
            </div>
          </div>
        )}

        {/* Tab 3: Upload & Predict & Pipeline Result */}
        {activeTab === 'upload' && (
          <div>
            {!analysisResult && !isAnalyzing && (
              <div style={{maxWidth: '680px', margin: '0 auto'}}>
                <div className="content-header">
                  <h1>Lancer une Analyse Diagnostique par IA</h1>
                </div>

                <div className="card glass">
                  <div className="form-group">
                    <label>Sélectionner le Patient</label>
                    <select 
                      className="form-control"
                      value={uploadPatientId}
                      onChange={(e) => setUploadPatientId(e.target.value)}
                    >
                      <option value="">-- Choisir un patient enregistré --</option>
                      {patients.map(p => (
                        <option key={p.id} value={p.id}>
                          {p.last_name.toUpperCase()} {p.first_name} (ID: {p.id})
                        </option>
                      ))}
                    </select>
                  </div>

                  <div className="form-group">
                    <label>Œil Ausculté</label>
                    <div style={{display: 'flex', gap: '16px', marginTop: '8px'}}>
                      <label style={{display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer'}}>
                        <input 
                          type="radio" 
                          name="eye" 
                          value="left" 
                          checked={uploadEye === 'left'}
                          onChange={() => setUploadEye('left')}
                        />
                        Œil Gauche
                      </label>
                      <label style={{display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer'}}>
                        <input 
                          type="radio" 
                          name="eye" 
                          value="right" 
                          checked={uploadEye === 'right'}
                          onChange={() => setUploadEye('right')}
                        />
                        Œil Droit
                      </label>
                    </div>
                  </div>

                  <div className="form-group" style={{marginTop: '24px'}}>
                    <label>Photographie du Fond d'Œil</label>
                    <div 
                      className="upload-dropzone"
                      onDragOver={handleDragOver}
                      onDrop={handleDrop}
                      onClick={() => fileInputRef.current.click()}
                    >
                      <Upload className="upload-icon" />
                      <p style={{fontWeight: 500}}>Glisser-déposer l'image ou cliquer pour parcourir</p>
                      <p style={{fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '8px'}}>Supporte les formats JPG, JPEG et PNG</p>
                      <input 
                        type="file" 
                        ref={fileInputRef} 
                        style={{display: 'none'}} 
                        onChange={handleFileChange}
                        accept="image/*"
                      />
                    </div>
                    
                    {previewUrl && (
                      <div className="image-preview-container">
                        <img src={previewUrl} alt="Aperçu" className="image-preview" />
                      </div>
                    )}
                  </div>

                  <button 
                    onClick={runOcularAnalysis}
                    className="btn btn-primary"
                    style={{marginTop: '20px'}}
                  >
                    <Activity size={16} /> Démarrer la Pipeline d'Analyse IA
                  </button>
                </div>
              </div>
            )}

            {/* In-progress analysis dashboard loader */}
            {isAnalyzing && (
              <div className="spinner-container" style={{minHeight: '400px'}}>
                <div className="spinner"></div>
                <h3 className="pulse-text" style={{fontSize: '1.2rem', fontWeight: 600}}>Analyse Diagnostique Rétinienne en Cours...</h3>
                <p style={{color: 'var(--accent-color)', fontWeight: 500, fontSize: '0.95rem'}}>{analysisStep}</p>
                <div style={{maxWidth: '400px', fontSize: '0.8rem', color: 'var(--text-muted)', textAlign: 'center', marginTop: '20px'}}>
                  Notre pipeline extrait les structures anatomiques, détecte 8 pathologies oculaires par classification multi-label et génère un overlay de chaleur de Grad-CAM avant de compiler le compte-rendu médical Mistral LLM.
                </div>
              </div>
            )}

            {/* Analysis results visualizer */}
            {analysisResult && (
              <div>
                <div className="content-header">
                  <div>
                    <h1 style={{fontSize: '1.75rem'}}>Rapport d'Analyse Diagnostique IA</h1>
                    <p style={{color: 'var(--text-secondary)', fontSize: '0.9rem', marginTop: '6px'}}>
                      Patient : <strong style={{color: '#fff'}}>{analysisResult.patient.last_name.toUpperCase()} {analysisResult.patient.first_name}</strong> | Œil : <strong style={{color: '#fff'}}>{analysisResult.image.eye === 'left' ? 'Gauche' : 'Droit'}</strong>
                    </p>
                  </div>
                  <button 
                    onClick={() => setAnalysisResult(null)} 
                    className="btn btn-secondary"
                    style={{width: 'auto'}}
                  >
                    Faire une nouvelle analyse
                  </button>
                </div>

                <div className="grid-2">
                  
                  {/* Left Column: Visualizations & CNN Probabilities */}
                  <div>
                    <div className="card glass">
                      <div className="card-title"><Eye size={20} /> Explicabilité Visuelle (Grad-CAM)</div>
                      <div className="comparison-grid">
                        <div className="image-card">
                          <p>Image Originale</p>
                          <div className="image-wrapper">
                            <img src={`${BACKEND_URL}/${analysisResult.image.filepath}`} alt="Original" />
                          </div>
                        </div>
                        <div className="image-card">
                          <p>Overlay de Chaleur Grad-CAM</p>
                          <div className="image-wrapper">
                            <img 
                              src={`${BACKEND_URL}/${analysisResult.image.gradcam_filepath}`} 
                              alt="Grad-CAM Overlay" 
                            />
                          </div>
                        </div>
                      </div>
                      <p style={{fontSize: '0.8rem', color: 'var(--text-secondary)', lineHeight: 1.5}}>
                        * L'overlay Grad-CAM met en surbrillance rouge et orange les zones d'attention sur lesquelles le réseau de neurones convolutif s'est basé pour réaliser ses prédictions cliniques.
                      </p>
                    </div>

                    <div className="card glass">
                      <div className="card-title"><Activity size={20} /> Probabilités de Classification Multi-Label</div>
                      <div className="prob-list">
                        {[
                          { key: 'Normal (N)', val: analysisResult.prediction.n_prob },
                          { key: 'Rétinopathie Diabétique (D)', val: analysisResult.prediction.d_prob },
                          { key: 'Glaucome (G)', val: analysisResult.prediction.g_prob },
                          { key: 'Cataracte (C)', val: analysisResult.prediction.c_prob },
                          { key: 'Dégénérescence Maculaire (A)', val: analysisResult.prediction.a_prob },
                          { key: 'Rétinopathie Hypertensive (H)', val: analysisResult.prediction.h_prob },
                          { key: 'Myopie Pathologique (M)', val: analysisResult.prediction.m_prob },
                          { key: 'Autre Anomalie / Lésion (O)', val: analysisResult.prediction.o_prob }
                        ].map((prob, idx) => {
                          const percentage = Math.round(prob.val * 100);
                          const isHigh = prob.val >= 0.5;

                          return (
                            <div className="prob-row" key={idx}>
                              <div className="prob-labels">
                                <span>{prob.key}</span>
                                <span style={{fontWeight: 700, color: isHigh ? 'var(--danger)' : 'var(--text-primary)'}}>{percentage}%</span>
                              </div>
                              <div className="prob-bar-bg">
                                <div 
                                  className={`prob-bar-fill ${isHigh ? 'high' : ''}`} 
                                  style={{width: `${percentage}%`}}
                                ></div>
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  </div>

                  {/* Right Column: Clinical LLM Report */}
                  <div className="card glass" style={{height: 'fit-content'}}>
                    <div className="card-title"><FileText size={20} /> Compte-rendu Médical Mistral LLM</div>
                    <div className="report-content">
                      {renderMarkdown(analysisResult.report.report_text)}
                    </div>
                  </div>

                </div>
              </div>
            )}
          </div>
        )}

        {/* Tab 4: History Table */}
        {activeTab === 'history' && (
          <div>
            <div className="content-header">
              <h1>Historique Médical Complet</h1>
            </div>

            <div className="card glass">
              {history.length === 0 ? (
                <p style={{color: 'var(--text-secondary)', textAlign: 'center', padding: '40px'}}>Aucun examen enregistré en base de données.</p>
              ) : (
                <table className="history-table">
                  <thead>
                    <tr>
                      <th>Date de l'examen</th>
                      <th>Patient</th>
                      <th>Sexe / Âge</th>
                      <th>Œil</th>
                      <th>Diagnostique Primaire (IA)</th>
                      <th>Rapport LLM</th>
                      <th>Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {history.map((item, idx) => {
                      const probs = item.prediction ? [
                        { name: 'Normal', val: item.prediction.n_prob },
                        { name: 'Diabète', val: item.prediction.d_prob },
                        { name: 'Glaucome', val: item.prediction.g_prob },
                        { name: 'Cataracte', val: item.prediction.c_prob },
                        { name: 'DMLA', val: item.prediction.a_prob },
                        { name: 'Hypertension', val: item.prediction.h_prob },
                        { name: 'Myopie', val: item.prediction.m_prob },
                        { name: 'Autre', val: item.prediction.o_prob }
                      ] : [];
                      
                      const topClass = probs.length > 0 ? probs.sort((a,b) => b.val - a.val)[0] : null;
                      
                      // Calculate age
                      const today = new Date();
                      const birth = new Date(item.patient.birth_date);
                      let age = today.getFullYear() - birth.getFullYear();
                      const m = today.getMonth() - birth.getMonth();
                      if (m < 0 || (m === 0 && today.getDate() < birth.getDate())) {
                          age--;
                      }

                      return (
                        <tr key={idx}>
                          <td>{new Date(item.image.uploaded_at).toLocaleString('fr-FR')}</td>
                          <td>
                            <strong style={{color: '#fff'}}>{item.patient.last_name.toUpperCase()}</strong> {item.patient.first_name}
                          </td>
                          <td>
                            {item.patient.gender === 'Female' ? 'F' : 'M'} / {age} ans
                          </td>
                          <td style={{textTransform: 'capitalize'}}>{item.image.eye === 'left' ? 'Gauche' : 'Droit'}</td>
                          <td>
                            {topClass ? (
                              <span className={topClass.name === 'Normal' ? 'tag tag-normal' : 'tag tag-pathology'}>
                                {topClass.name} ({Math.round(topClass.val * 100)}%)
                              </span>
                            ) : 'Incomplète'}
                          </td>
                          <td>
                            {item.report ? (
                              <span style={{color: 'var(--success)', display: 'inline-flex', alignItems: 'center', gap: '4px', fontSize: '0.85rem', fontWeight: 500}}>
                                <CheckCircle size={14} /> Prêt
                              </span>
                            ) : (
                              <span style={{color: 'var(--warning)', display: 'inline-flex', alignItems: 'center', gap: '4px', fontSize: '0.85rem', fontWeight: 500}}>
                                <AlertTriangle size={14} /> Manquant
                              </span>
                            )}
                          </td>
                          <td>
                            <button 
                              onClick={() => viewHistoryExam(item)}
                              className="btn btn-secondary" 
                              style={{padding: '6px 12px', fontSize: '0.8rem', width: 'auto'}}
                            >
                              <Eye size={14} /> Ouvrir le dossier
                            </button>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              )}
            </div>
          </div>
        )}

      </div>
    </div>
  );
}

export default App;
