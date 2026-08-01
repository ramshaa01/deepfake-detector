import { useState, useRef, useEffect } from 'react';
import { UploadCloud, Image as ImageIcon, AlertCircle, CheckCircle2, XCircle, Info, Loader2, ChevronDown, ChevronUp } from 'lucide-react';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs) {
  return twMerge(clsx(inputs));
}

export default function App() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [coldStartNotice, setColdStartNotice] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [showInfo, setShowInfo] = useState(false);
  const fileInputRef = useRef(null);

  const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8001';

  // Handle Cold Start Warning Timer
  useEffect(() => {
    let timer;
    if (loading) {
      timer = setTimeout(() => {
        setColdStartNotice(true);
      }, 5000); // 5 seconds
    } else {
      setColdStartNotice(false);
    }
    return () => clearTimeout(timer);
  }, [loading]);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) processFile(selectedFile);
  };

  const processFile = (selectedFile) => {
    setError(null);
    setResult(null);

    // Validate type
    if (!selectedFile.type.startsWith('image/')) {
      setError('Please upload a valid image file.');
      return;
    }

    // Validate size (10MB limit)
    if (selectedFile.size > 10 * 1024 * 1024) {
      setError('File is too large. Maximum size is 10MB.');
      return;
    }

    setFile(selectedFile);
    setPreviewUrl(URL.createObjectURL(selectedFile));
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    const droppedFile = e.dataTransfer.files?.[0];
    if (droppedFile) processFile(droppedFile);
  };

  const analyzeImage = async () => {
    if (!file) return;

    setLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}));
        throw new Error(errorData.detail || `Server error (${res.status})`);
      }

      const data = await res.json();
      setResult(data);
    } catch (err) {
      setError(err.message || 'An error occurred during analysis.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen p-6 md:p-12 flex flex-col items-center">
      <div className="w-full max-w-3xl space-y-8">
        
        {/* Header */}
        <header className="text-center space-y-4">
          <div className="inline-flex items-center justify-center p-3 bg-indigo-500/10 rounded-2xl mb-2 ring-1 ring-indigo-500/30">
            <ScanFaceIcon className="w-8 h-8 text-indigo-400" />
          </div>
          <h1 className="text-4xl md:text-5xl font-extrabold tracking-tight text-transparent bg-clip-text bg-gradient-to-r from-indigo-300 via-purple-300 to-indigo-300">
            AI Face Detector
          </h1>
          <p className="text-slate-400 text-lg max-w-xl mx-auto">
            Upload a portrait to determine if it's a genuine photograph or a synthetic AI-generated face (StyleGAN2).
          </p>
        </header>

        {/* Info Accordion */}
        <div className="bg-slate-900/50 border border-slate-800 rounded-2xl overflow-hidden backdrop-blur-sm">
          <button 
            onClick={() => setShowInfo(!showInfo)}
            className="w-full flex items-center justify-between p-4 text-slate-300 hover:text-white hover:bg-slate-800/50 transition-colors"
          >
            <div className="flex items-center gap-2 font-medium">
              <Info className="w-5 h-5 text-indigo-400" />
              How this works & Known Limitations
            </div>
            {showInfo ? <ChevronUp className="w-5 h-5" /> : <ChevronDown className="w-5 h-5" />}
          </button>
          
          {showInfo && (
            <div className="p-4 pt-0 text-sm text-slate-400 border-t border-slate-800/50 leading-relaxed space-y-3">
              <p>
                This tool uses an EfficientNet-B0 Convolutional Neural Network (CNN) fused with Frequency-Domain (FFT) features to detect synthetic artefacts left by AI generators like StyleGAN2. It analyzes both the spatial pixels and the underlying frequency spectrum of the face.
              </p>
              <ul className="list-disc pl-5 space-y-1">
                <li><strong className="text-slate-300">Eyeglasses Bias:</strong> The model currently has a known shortcut associating eyeglasses with synthetic faces due to imbalances in the training data.</li>
                <li><strong className="text-slate-300">Image Degradation:</strong> Heavy compression, extreme downscaling, or heavy blur will destroy the high-frequency artefacts the model relies on, causing it to default to predicting "Fake".</li>
                <li><strong className="text-slate-300">Scope:</strong> This detects completely synthesized faces (StyleGAN), not video deepfake manipulations (e.g., FaceSwap).</li>
              </ul>
            </div>
          )}
        </div>

        {/* Upload Zone */}
        <div 
          className={cn(
            "relative group border-2 border-dashed rounded-3xl p-8 md:p-12 transition-all duration-200 ease-out flex flex-col items-center justify-center text-center cursor-pointer bg-slate-900/30",
            isDragging ? "border-indigo-500 bg-indigo-500/10 scale-[1.01]" : "border-slate-700 hover:border-slate-500 hover:bg-slate-800/30",
            file && "border-slate-700 border-solid bg-slate-900/50"
          )}
          onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={handleDrop}
          onClick={() => !file && fileInputRef.current?.click()}
        >
          <input 
            type="file" 
            ref={fileInputRef} 
            onChange={handleFileChange} 
            accept="image/*" 
            className="hidden" 
          />
          
          {file ? (
            <div className="w-full space-y-6">
              <div className="relative w-48 h-48 mx-auto rounded-2xl overflow-hidden ring-4 ring-slate-800 shadow-2xl">
                <img src={previewUrl} alt="Preview" className="w-full h-full object-cover" />
                <button 
                  onClick={(e) => { e.stopPropagation(); setFile(null); setResult(null); setError(null); }}
                  className="absolute top-2 right-2 p-1.5 bg-black/60 hover:bg-red-500 text-white rounded-full backdrop-blur-md transition-colors"
                >
                  <XCircle className="w-5 h-5" />
                </button>
              </div>
              
              <div className="space-y-4">
                <p className="text-slate-300 font-medium truncate px-4">{file.name}</p>
                
                {!result && !loading && (
                  <button 
                    onClick={(e) => { e.stopPropagation(); analyzeImage(); }}
                    className="w-full md:w-auto px-8 py-3 bg-indigo-600 hover:bg-indigo-500 text-white rounded-xl font-semibold shadow-lg shadow-indigo-500/20 transition-all hover:-translate-y-0.5"
                  >
                    Analyze Image
                  </button>
                )}
              </div>
            </div>
          ) : (
            <div className="space-y-4 pointer-events-none">
              <div className="w-20 h-20 mx-auto bg-slate-800 rounded-full flex items-center justify-center group-hover:scale-110 transition-transform duration-300 shadow-xl">
                <UploadCloud className="w-10 h-10 text-indigo-400" />
              </div>
              <div className="space-y-1">
                <h3 className="text-xl font-semibold text-slate-200">Drag & drop a face image</h3>
                <p className="text-slate-500">or click to browse files</p>
              </div>
              <p className="text-xs text-slate-600 font-medium pt-2">Supports JPG, PNG, WEBP up to 10MB</p>
            </div>
          )}
        </div>

        {/* Loading State */}
        {loading && (
          <div className="bg-slate-900/50 border border-slate-800 rounded-2xl p-8 flex flex-col items-center justify-center text-center space-y-4 backdrop-blur-sm shadow-xl animate-in fade-in duration-300">
            <Loader2 className="w-10 h-10 text-indigo-500 animate-spin" />
            <div className="space-y-2">
              <p className="text-lg font-medium text-slate-200">Analyzing Face Data...</p>
              
              {coldStartNotice ? (
                <div className="inline-flex items-start gap-2 text-amber-400 bg-amber-400/10 px-4 py-3 rounded-xl text-sm max-w-sm text-left animate-in slide-in-from-bottom-2">
                  <AlertCircle className="w-5 h-5 shrink-0 mt-0.5" />
                  <p>Waking up the server. This may take up to a minute on the first request since the free-tier backend sleeps when inactive.</p>
                </div>
              ) : (
                <p className="text-sm text-slate-500">Extracting features and generating heatmap</p>
              )}
            </div>
          </div>
        )}

        {/* Error State */}
        {error && !loading && (
          <div className="bg-red-500/10 border border-red-500/20 rounded-2xl p-6 flex items-start gap-4 animate-in fade-in slide-in-from-top-4">
            <AlertCircle className="w-6 h-6 text-red-400 shrink-0 mt-0.5" />
            <div>
              <h3 className="text-lg font-semibold text-red-400 mb-1">Analysis Failed</h3>
              <p className="text-slate-300 leading-relaxed">{error}</p>
            </div>
          </div>
        )}

        {/* Results State */}
        {result && !loading && (
          <div className="bg-slate-900/80 border border-slate-700 rounded-3xl p-6 md:p-8 shadow-2xl backdrop-blur-md animate-in fade-in slide-in-from-bottom-8">
            <h2 className="text-2xl font-bold text-center text-slate-200 mb-8">Analysis Results</h2>
            
            <div className="grid md:grid-cols-2 gap-8 items-start">
              
              {/* Score Card */}
              <div className="space-y-6">
                <div className={cn(
                  "p-6 rounded-2xl border flex flex-col items-center justify-center text-center space-y-2 relative overflow-hidden",
                  result.label === 'real' 
                    ? "bg-emerald-500/10 border-emerald-500/30 text-emerald-400" 
                    : "bg-rose-500/10 border-rose-500/30 text-rose-400"
                )}>
                  {/* Subtle background glow */}
                  <div className={cn(
                    "absolute -inset-10 opacity-20 blur-2xl",
                    result.label === 'real' ? "bg-emerald-500" : "bg-rose-500"
                  )} />
                  
                  <span className="text-sm font-semibold uppercase tracking-wider relative">Prediction</span>
                  <span className="text-5xl font-black capitalize relative">
                    {result.label}
                  </span>
                  <div className="flex items-center gap-2 pt-2 relative">
                    {result.label === 'real' ? <CheckCircle2 className="w-5 h-5" /> : <AlertCircle className="w-5 h-5" />}
                    <span className="font-medium text-lg">{(result.confidence * 100).toFixed(1)}% Confidence</span>
                  </div>
                </div>

                <div className="bg-slate-800/50 rounded-2xl p-5 border border-slate-700/50 space-y-4">
                  <div className="flex justify-between items-center text-sm">
                    <span className="text-slate-400">Synthetic Probability</span>
                    <span className="font-mono text-slate-200">{(result.probability_fake * 100).toFixed(2)}%</span>
                  </div>
                  <div className="h-2.5 w-full bg-slate-900 rounded-full overflow-hidden flex">
                    <div 
                      className="bg-emerald-500 transition-all duration-1000 ease-out" 
                      style={{ width: `${(1 - result.probability_fake) * 100}%` }}
                    />
                    <div 
                      className="bg-rose-500 transition-all duration-1000 ease-out" 
                      style={{ width: `${result.probability_fake * 100}%` }}
                    />
                  </div>
                  <div className="flex justify-between items-center text-xs text-slate-500 font-medium pt-1">
                    <span>Real</span>
                    <span>Fake</span>
                  </div>
                </div>

                <div className="text-center text-sm text-slate-500">
                  Inference time: <span className="font-mono text-slate-300">{result.inference_time_ms}ms</span>
                </div>
              </div>

              {/* Heatmap Card */}
              <div className="bg-slate-800/50 rounded-2xl p-5 border border-slate-700/50 flex flex-col items-center">
                <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-4 flex items-center gap-2">
                  <ImageIcon className="w-4 h-4" />
                  Grad-CAM Heatmap
                </h3>
                
                {result.heatmap_base64 ? (
                  <div className="relative w-full max-w-[224px] aspect-square rounded-xl overflow-hidden ring-1 ring-white/10 shadow-lg">
                    <img 
                      src={`data:image/jpeg;base64,${result.heatmap_base64}`} 
                      alt="Grad-CAM Heatmap" 
                      className="w-full h-full object-cover"
                    />
                  </div>
                ) : (
                  <div className="w-full max-w-[224px] aspect-square rounded-xl border border-dashed border-slate-600 flex flex-col items-center justify-center text-slate-500 p-4 text-center">
                    <ImageIcon className="w-8 h-8 mb-2 opacity-50" />
                    <span className="text-sm">Heatmap not available for this image</span>
                  </div>
                )}
                
                <p className="text-xs text-slate-500 mt-4 text-center">
                  Red areas indicate regions the model focused on to make its decision.
                </p>
              </div>

            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function ScanFaceIcon(props) {
  return (
    <svg
      {...props}
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M3 7V5a2 2 0 0 1 2-2h2" />
      <path d="M17 3h2a2 2 0 0 1 2 2v2" />
      <path d="M21 17v2a2 2 0 0 1-2 2h-2" />
      <path d="M7 21H5a2 2 0 0 1-2-2v-2" />
      <path d="M8 14s1.5 2 4 2 4-2 4-2" />
      <path d="M9 9h.01" />
      <path d="M15 9h.01" />
    </svg>
  );
}
