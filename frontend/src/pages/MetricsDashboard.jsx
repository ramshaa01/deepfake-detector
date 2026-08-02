import metrics from '../data/model_metrics.json';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';
import { AlertTriangle, ExternalLink, Info } from 'lucide-react';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer,
  ReferenceLine, CartesianGrid, Cell,
} from 'recharts';

function cn(...inputs) { return twMerge(clsx(inputs)); }

// ── Stat Card ──────────────────────────────────────────────────────────────
function StatCard({ label, value, highlight = false }) {
  return (
    <div className={cn(
      "bg-slate-900 border rounded-2xl p-5 flex flex-col items-center text-center",
      highlight ? "border-indigo-500/40 ring-1 ring-indigo-500/20" : "border-slate-800"
    )}>
      <span className="text-xs font-semibold uppercase tracking-wider text-slate-500 mb-2">{label}</span>
      <span className={cn(
        "text-3xl font-black tabular-nums",
        highlight ? "text-indigo-400" : "text-slate-100"
      )}>{value}</span>
    </div>
  );
}

// ── Confusion Matrix ────────────────────────────────────────────────────────
function ConfusionMatrix() {
  const { matrix, labels } = metrics.confusion_matrix;
  const total = matrix.flat().reduce((a, b) => a + b, 0);
  const colors = [
    // TP (correct real), FN
    ['bg-emerald-500/30 text-emerald-300 border-emerald-500/30', 'bg-rose-500/20 text-rose-300 border-rose-500/20'],
    // FP, TN (correct fake)
    ['bg-rose-500/20 text-rose-300 border-rose-500/20', 'bg-indigo-500/30 text-indigo-300 border-indigo-500/30'],
  ];
  const cellLabel = [
    ['True Real', 'False Fake (FN)'],
    ['False Real (FP)', 'True Fake'],
  ];

  return (
    <div className="flex flex-col items-center gap-3">
      {/* Column headers */}
      <div className="grid grid-cols-[120px_1fr_1fr] gap-2 w-full max-w-sm text-center">
        <div />
        {labels.map(l => (
          <div key={l} className="text-xs font-semibold uppercase tracking-wider text-slate-400">
            Pred: {l}
          </div>
        ))}
      </div>
      {matrix.map((row, ri) => (
        <div key={ri} className="grid grid-cols-[120px_1fr_1fr] gap-2 w-full max-w-sm items-center">
          <div className="text-xs font-semibold uppercase tracking-wider text-slate-400 text-right pr-2">
            True: {labels[ri]}
          </div>
          {row.map((val, ci) => (
            <div
              key={ci}
              className={cn(
                "rounded-xl border p-4 flex flex-col items-center justify-center aspect-square",
                colors[ri][ci]
              )}
              title={cellLabel[ri][ci]}
            >
              <span className="text-2xl font-black">{val}</span>
              <span className="text-xs opacity-70 mt-1">{((val / total) * 100).toFixed(1)}%</span>
              <span className="text-xs opacity-60 mt-1 hidden sm:block">{cellLabel[ri][ci]}</span>
            </div>
          ))}
        </div>
      ))}
      <p className="text-xs text-slate-600 text-center mt-1">Rows = True class · Columns = Predicted class</p>
    </div>
  );
}

// ── Custom Tooltip ─────────────────────────────────────────────────────────
function RobustnessTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-slate-900 border border-slate-700 rounded-xl p-3 text-sm shadow-xl">
      <p className="font-semibold text-slate-200 mb-2">{label}</p>
      {payload.map(p => (
        <p key={p.name} style={{ color: p.color }}>
          {p.name}: <span className="font-mono font-bold">{p.value.toFixed(1)}%</span>
        </p>
      ))}
    </div>
  );
}

// ── Family color mapping ───────────────────────────────────────────────────
const FAMILY_COLOR = {
  baseline: '#818cf8',
  jpeg:     '#f59e0b',
  blur:     '#f43f5e',
  resize:   '#06b6d4',
};

// ── Main Dashboard ─────────────────────────────────────────────────────────
export default function MetricsDashboard() {
  const { models, robustness } = metrics;
  const fusion = models.fusion;
  const baseline = models.baseline;

  // Comparison bar chart data
  const compareData = [
    { metric: 'Accuracy',  baseline: (baseline.accuracy * 100).toFixed(1),  fusion: (fusion.accuracy * 100).toFixed(1) },
    { metric: 'Precision', baseline: (baseline.precision * 100).toFixed(1), fusion: (fusion.precision * 100).toFixed(1) },
    { metric: 'Recall',    baseline: (baseline.recall * 100).toFixed(1),    fusion: (fusion.recall * 100).toFixed(1) },
    { metric: 'F1',        baseline: (baseline.f1 * 100).toFixed(1),        fusion: (fusion.f1 * 100).toFixed(1) },
    { metric: 'ROC-AUC',   baseline: (baseline.roc_auc * 100).toFixed(1),   fusion: (fusion.roc_auc * 100).toFixed(1) },
  ];

  return (
    <div className="min-h-screen p-4 md:p-12 flex flex-col items-center">
      <div className="w-full max-w-3xl space-y-10">

        {/* Header */}
        <div className="text-center space-y-3">
          <h1 className="text-3xl md:text-4xl font-extrabold tracking-tight text-slate-100">
            Model Performance Report
          </h1>
          <p className="text-slate-400 max-w-xl mx-auto">
            Results from the fine-tuned EfficientNet-B0 + FFT Fusion model evaluated on a
            held-out test set (300 images, balanced).
          </p>
          <div className="inline-flex items-center gap-2 bg-slate-800/60 border border-slate-700 px-4 py-2 rounded-full text-xs text-slate-400">
            <Info className="w-3.5 h-3.5 text-indigo-400" />
            Metrics from held-out test set evaluation — see{' '}
            <a
              href="https://github.com/ramshaa01/deepfake-detector"
              target="_blank"
              rel="noopener noreferrer"
              className="text-indigo-400 hover:underline inline-flex items-center gap-1"
            >
              GitHub repo <ExternalLink className="w-3 h-3" />
            </a>{' '}
            for full methodology.
          </div>
        </div>

        {/* ── Final Model: Stat Cards ── */}
        <section className="space-y-4">
          <h2 className="text-lg font-bold text-slate-300">Fusion Model — Final Metrics</h2>
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-3">
            <StatCard label="Accuracy"  value={`${(fusion.accuracy  * 100).toFixed(2)}%`} highlight />
            <StatCard label="ROC-AUC"   value={fusion.roc_auc.toFixed(4)}                  highlight />
            <StatCard label="Precision" value={fusion.precision.toFixed(4)} />
            <StatCard label="Recall"    value={fusion.recall.toFixed(4)}    />
            <StatCard label="F1 Score"  value={fusion.f1.toFixed(4)}        />
          </div>
          <p className="text-xs text-slate-600 text-right">
            Avg inference: {fusion.inference_ms} ms/image · {fusion.name}
          </p>
        </section>

        {/* ── Model Comparison Bar Chart ── */}
        <section className="space-y-4">
          <h2 className="text-lg font-bold text-slate-300">Model Comparison (% score)</h2>
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6">
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={compareData} barCategoryGap="30%" barGap={4}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                <XAxis dataKey="metric" tick={{ fill: '#94a3b8', fontSize: 12 }} axisLine={false} tickLine={false} />
                <YAxis domain={[60, 100]} tickFormatter={v => `${v}%`} tick={{ fill: '#94a3b8', fontSize: 11 }} axisLine={false} tickLine={false} />
                <Tooltip
                  content={({ active, payload, label }) => {
                    if (!active || !payload?.length) return null;
                    return (
                      <div className="bg-slate-900 border border-slate-700 rounded-xl p-3 text-sm shadow-xl">
                        <p className="font-semibold text-slate-200 mb-2">{label}</p>
                        {payload.map(p => (
                          <p key={p.name} style={{ color: p.color }}>
                            {p.name}: <span className="font-mono font-bold">{p.value}%</span>
                          </p>
                        ))}
                      </div>
                    );
                  }}
                />
                <Legend wrapperStyle={{ paddingTop: '16px', fontSize: '12px', color: '#94a3b8' }} />
                <Bar dataKey="baseline" name="CNN-only (Day 10)" fill="#475569" radius={[4, 4, 0, 0]} />
                <Bar dataKey="fusion"   name="CNN+FFT Fusion (Day 16)" fill="#6366f1" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </section>

        {/* ── Confusion Matrix ── */}
        <section className="space-y-4">
          <h2 className="text-lg font-bold text-slate-300">Confusion Matrix — Fusion Model</h2>
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 flex flex-col items-center">
            <ConfusionMatrix />
          </div>
        </section>

        {/* ── Robustness Chart ── */}
        <section className="space-y-4">
          <h2 className="text-lg font-bold text-slate-300">Robustness Under Perturbations</h2>
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6">
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={robustness} barSize={20} barCategoryGap="20%">
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                <XAxis
                  dataKey="condition"
                  tick={{ fill: '#94a3b8', fontSize: 10 }}
                  axisLine={false}
                  tickLine={false}
                  angle={-30}
                  textAnchor="end"
                  height={60}
                />
                <YAxis
                  yAxisId="acc"
                  domain={[0, 100]}
                  tickFormatter={v => `${v}%`}
                  tick={{ fill: '#94a3b8', fontSize: 11 }}
                  axisLine={false}
                  tickLine={false}
                  label={{ value: 'Accuracy %', angle: -90, position: 'insideLeft', fill: '#64748b', fontSize: 11, dx: -4 }}
                />
                <Tooltip content={<RobustnessTooltip />} />
                <ReferenceLine yAxisId="acc" y={79.33} stroke="#818cf8" strokeDasharray="4 4" label={{ value: 'Clean baseline', fill: '#818cf8', fontSize: 10, position: 'insideTopRight' }} />
                <Bar yAxisId="acc" dataKey="accuracy" name="Accuracy" radius={[4, 4, 0, 0]}>
                  {robustness.map((entry, i) => (
                    <Cell key={i} fill={FAMILY_COLOR[entry.family]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <div className="flex flex-wrap gap-4 mt-4 justify-center text-xs">
              {Object.entries(FAMILY_COLOR).map(([family, color]) => (
                <div key={family} className="flex items-center gap-1.5">
                  <div className="w-3 h-3 rounded-sm" style={{ background: color }} />
                  <span className="text-slate-400 capitalize">{family === 'baseline' ? 'Clean baseline' : family}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Per-class breakdown */}
          <div className="bg-slate-900/50 border border-slate-800 rounded-2xl overflow-x-auto">
            <table className="w-full text-sm min-w-[540px]">
              <thead>
                <tr className="border-b border-slate-800">
                  {['Condition', 'Accuracy', 'ROC-AUC', 'Retention %', 'Real Acc', 'Fake Acc'].map(h => (
                    <th key={h} className="text-left px-4 py-3 text-xs font-semibold uppercase tracking-wider text-slate-500">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {robustness.map((r, i) => (
                  <tr key={i} className={cn("border-b border-slate-800/50 transition-colors hover:bg-slate-800/30", i === 0 && "bg-indigo-500/5")}>
                    <td className="px-4 py-3 font-medium text-slate-300">{r.condition}</td>
                    <td className="px-4 py-3 font-mono text-slate-200">{r.accuracy.toFixed(2)}%</td>
                    <td className="px-4 py-3 font-mono text-slate-400">{r.roc_auc.toFixed(4)}</td>
                    <td className={cn("px-4 py-3 font-mono font-semibold", r.retention >= 90 ? "text-emerald-400" : r.retention >= 75 ? "text-amber-400" : "text-rose-400")}>
                      {r.retention.toFixed(1)}%
                    </td>
                    <td className="px-4 py-3 font-mono text-slate-400">{r.real_acc.toFixed(2)}%</td>
                    <td className="px-4 py-3 font-mono text-slate-400">{r.fake_acc.toFixed(2)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        {/* ── Known Limitations ── */}
        <section className="space-y-4">
          <h2 className="text-lg font-bold text-slate-300">Known Limitations</h2>
          <div className="bg-amber-500/5 border border-amber-500/20 rounded-2xl p-6 space-y-4">
            <div className="flex items-start gap-3">
              <AlertTriangle className="w-5 h-5 text-amber-400 shrink-0 mt-0.5" />
              <div className="space-y-3">
                <div>
                  <h3 className="font-semibold text-amber-300 mb-1">Eyeglasses False-Positive Bias</h3>
                  <p className="text-sm text-slate-400 leading-relaxed">
                    The model has a documented spurious correlation between eyeglasses and "Fake" predictions,
                    caused by a distribution mismatch between the FFHQ (real) and StyleGAN2 (fake) training sets.
                    People wearing glasses may be incorrectly flagged as AI-generated.
                  </p>
                </div>
                <div>
                  <h3 className="font-semibold text-amber-300 mb-1">Catastrophic Collapse Under Blur / Heavy Downscaling</h3>
                  <p className="text-sm text-slate-400 leading-relaxed">
                    The fusion model relies on high-frequency spectral artefacts produced by GAN generators.
                    Heavy Gaussian blur (σ≥2) or extreme downscaling (0.25×) destroys those artefacts, causing
                    <strong className="text-slate-300"> Real accuracy to collapse to 10–23%</strong> while Fake accuracy stays high.
                    The retention chart above shows this drop clearly.
                  </p>
                </div>
                <a
                  href="https://github.com/ramshaa01/deepfake-detector#known-limitations"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-1 text-sm text-indigo-400 hover:underline"
                >
                  Full discussion in README <ExternalLink className="w-3.5 h-3.5" />
                </a>
              </div>
            </div>
          </div>
        </section>

      </div>
    </div>
  );
}
