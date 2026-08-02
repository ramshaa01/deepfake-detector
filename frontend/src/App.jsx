import { BrowserRouter, Routes, Route } from 'react-router-dom';
import NavBar from './components/NavBar';
import DetectorPage from './pages/DetectorPage';
import MetricsDashboard from './pages/MetricsDashboard';

export default function App() {
  return (
    <BrowserRouter>
      <NavBar />
      <main>
        <Routes>
          <Route path="/"        element={<DetectorPage />} />
          <Route path="/metrics" element={<MetricsDashboard />} />
        </Routes>
      </main>
    </BrowserRouter>
  );
}
