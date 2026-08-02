import { NavLink, useLocation } from 'react-router-dom';
import { ScanSearch, BarChart3 } from 'lucide-react';

export default function NavBar() {
  return (
    <nav className="sticky top-0 z-50 w-full bg-slate-950/80 backdrop-blur-md border-b border-slate-800">
      <div className="max-w-3xl mx-auto px-4 md:px-12 flex items-center justify-between h-14">
        <span className="font-bold text-slate-200 tracking-tight">AI Face Detector</span>
        <div className="flex items-center gap-1">
          <NavLink
            to="/"
            className={({ isActive }) =>
              `flex items-center gap-2 px-4 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                isActive
                  ? 'bg-indigo-600 text-white'
                  : 'text-slate-400 hover:text-white hover:bg-slate-800'
              }`
            }
          >
            <ScanSearch className="w-4 h-4" />
            Detector
          </NavLink>
          <NavLink
            to="/metrics"
            className={({ isActive }) =>
              `flex items-center gap-2 px-4 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                isActive
                  ? 'bg-indigo-600 text-white'
                  : 'text-slate-400 hover:text-white hover:bg-slate-800'
              }`
            }
          >
            <BarChart3 className="w-4 h-4" />
            Model Metrics
          </NavLink>
        </div>
      </div>
    </nav>
  );
}
