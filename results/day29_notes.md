# Day 29 Deployment Notes — Frontend to Vercel + End-to-End Verification

## Summary

Day 29 deployed the React frontend to Vercel and ran a full end-to-end verification
against both live services. Both the Render backend and Vercel frontend are independently
deployed and fully operational.

| Service | URL | Status |
|---|---|---|
| **Backend API** | https://deepfake-detector-k62g.onrender.com | Live (Render free tier) |
| **Frontend** | https://deepfake-detector-zeta.vercel.app | Live (Vercel Hobby tier) |

---

## Vercel Deployment Steps

1. Visited [vercel.com](https://vercel.com), logged in with GitHub (`ramshaa01`).
2. **Add New → Project → Import** `ramshaa01/deepfake-detector`.
3. Configuration set:
   - **Root Directory:** `frontend` (critical — not the repo root)
   - **Framework Preset:** Vite (auto-detected)
   - **Build Command:** `npm run build`
   - **Output Directory:** `dist`
4. **Environment Variable** added in Vercel dashboard:
   - `VITE_API_URL` = `https://deepfake-detector-k62g.onrender.com`
5. Clicked **Deploy**. Build completed in ~60 seconds.
6. Assigned domain: `deepfake-detector-zeta.vercel.app`

### Pre-deployment fix committed
A `frontend/vercel.json` file was added before deployment:
```json
{ "rewrites": [{ "source": "/(.*)", "destination": "/index.html" }] }
```
This is required for React Router's `/metrics` route to work on direct URL access —
without it, Vercel's CDN returns a 404 on any path except `/`.

---

## End-to-End Verification Results

All tests run against the live Render backend from the client machine (India → Render
US East). Timestamp: 2026-08-04.

| Test | Result | Round-Trip | Notes |
|---|---|---|---|
| `GET /` (Vercel — Detector page) | **PASS** | 843 ms | 200 OK, text/html, 458 bytes |
| `GET /metrics` (Vercel — SPA route) | **PASS** | 1,105 ms | vercel.json rewrite confirmed working |
| `GET /health` (Render) | **PASS** | 62,383 ms | **Cold start — Render was idle, woke from sleep** |
| `POST /predict` — REAL image | **PASS** | 2,563 ms | label=real, conf=77.4%, heatmap=11,456 chars |
| `POST /predict` — FAKE image | **PASS** | 2,264 ms | label=fake, conf=97.1%, heatmap=11,748 chars |
| `POST /predict` — No-face blank | **PASS** | 1,481 ms | 400 Bad Request, correct error message |

**Overall: ALL PASS (6/6)**

---

## Latency Analysis

### Cold Start
The `/health` check took **62.4 seconds** — Render had idled since the last session.
This validates the frontend's 5-second cold-start warning timer as essential UX. After
the container woke, subsequent requests were fast.

### Backend-Only Inference Latency
| Image | Backend inference_time_ms (reported by server) |
|---|---|
| Real (00022.jpg) | 304.51 ms |
| Fake (01HYNP6M67.jpg) | 207.78 ms |
| **Average** | **~256 ms** |

This is consistent with the Day 16-17 corrected single-image benchmark (~241 ms)
and within the expected ±30 ms variance from CPU load differences on the free-tier container.

### Client Round-Trip Latency
| | Real | Fake | Average |
|---|---|---|---|
| Round-trip (client → Render → client) | 2,563 ms | 2,264 ms | **2,413 ms** |

The ~2.4 second round-trip is **network-dominated** (India → Render US East), not
compute-dominated. Backend processing accounts for only ~256 ms (~11%) of the total
round-trip. For users geographically closer to Render's US East region, round-trip will
be substantially lower.

**Resume line clarification:** The ~241 ms figure in the README resume line refers to
**backend-only CPU inference latency** (the number relevant to production scaling decisions),
not the full client round-trip which is network-topology-dependent.

---

## Prediction Consistency (vs. Day 24 Live Verification)

| Image | Day 24 (Post-Day 24 fix) | Day 29 | Match |
|---|---|---|---|
| REAL (00022.jpg) | label=real, conf=0.7738 | label=real, conf=0.7738 | ✓ Identical |
| FAKE (01HYNP6M67.jpg) | label=fake, conf=0.9708 | label=fake, conf=0.9708 | ✓ Identical |
| Blank (no face) | 400 Bad Request | 400 Bad Request | ✓ Identical |

Model outputs are deterministic and stable across sessions (CPU inference, no sampling).

---

## Cold-Start UX Test

The `/health` call at the start of the session took 62 seconds (Render idle → cold
start). This means a user visiting the live site immediately after a period of inactivity
would see the "Waking up the server" notice appear after the 5-second timer fires.
The frontend cold-start handling (Day 25/26) is working as intended.
