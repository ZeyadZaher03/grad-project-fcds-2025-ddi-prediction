# DDI Mobile App

React Native (Expo) client for drug interaction lookup. Users can search for drugs, select any two, and view an animated probability of a potential interaction returned by the backend API.

## Requirements
- Node.js 18+
- Expo CLI (`npx expo` is installed automatically through npm)
- Running backend API providing `/v1/drugs/search` and `/v1/ddi/predict` endpoints (defaults to `http://localhost:8080`)

## Getting Started
1. Install dependencies (already installed by scaffolding, run again if needed):
   ```bash
   npm install
   ```
2. Update the `API_BASE` constant in `App.js` if your backend runs on a different host (e.g. LAN IP or tunnel).
3. Start the Expo dev server:
   ```bash
   npm run start
   ```
4. Launch on your preferred platform:
   - Press `a` for Android emulator
   - Press `i` for iOS simulator
   - Press `w` for web preview

## Usage
- Enter at least three characters to search for a drug name.
- Tap to select up to two results; the primary button becomes active once two drugs are chosen.
- The results screen animates the probability bar and labels the interaction risk.
- Tap **Check another pair** to return to search.

## Notes
- When testing on a physical device, replace `http://localhost:8080` with your machine's LAN IP or an HTTPS tunnel URL.
- The interaction label mirrors backend thresholds (`>=0.7` likely, `>=0.4` uncertain, else unlikely).
