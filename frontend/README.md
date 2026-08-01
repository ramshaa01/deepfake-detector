# Deepfake Detector Frontend

This directory contains the React frontend for the AI-Generated Face Detector, built with Vite and Tailwind CSS.

## Setup

```bash
cd frontend
npm install
```

## Running Locally

To run the frontend against the live Render production backend:
```bash
# This uses the default VITE_API_URL in .env
npm run dev
```

To run the frontend against your local backend:
1. Create a `.env.local` file (or edit `.env`).
2. Add: `VITE_API_URL=http://127.0.0.1:8000` (or whichever port you use locally).
3. Run `npm run dev`.

## Build for Production

```bash
npm run build
npm run preview
```

## Features
- **Drag & Drop Upload:** Client-side validation for image files up to 10MB.
- **Cold-Start Handling:** Displays a clear warning message if the server request takes longer than 5 seconds, explaining that the free-tier backend is waking up.
- **Results Display:** Clear visual styling for real vs fake predictions, along with confidence bars and Grad-CAM heatmap visualization.
- **Error Handling:** Displays 400 (no-face-detected) and 500 errors gracefully to the user.
