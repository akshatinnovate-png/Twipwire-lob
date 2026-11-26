# TRIPWIRE-LOB desktop app

A downloadable app for Windows, macOS and Linux. It includes the Python engine, so nothing else needs to be installed.

* **Cockpit:** the live engine. Pick a data source (synthetic, AAPL/AMZN/MSFT real days, live BTC-USDT) and inject shocks with keys 1/2/3/0. The **DESK** window shows the trust guard and three market-making desks side by side.
* **Desk menu:** start or stop the **live paper desk** on Binance BTC/ETH books, open the **track record** report, and edit the events calendar.
* **Data menu:** download the LOBSTER samples and build their caches, record the BTC book, and download crypto history (free Tardis days).
* **Benchmarks menu:** re-run the desk, crypto and forecasting benchmarks and the stage preflight check.

Data and results are stored in `Documents/TripwireLOB`. Only paper trading is supported: no order ever leaves the machine.

## Download

Push a tag such as `v1.0.0`. GitHub Actions (`.github/workflows/desktop.yml`) then builds the installers and attaches them to the release. You can also start a manual run from the Actions tab; its installers are saved as workflow artifacts.

## Build locally

```bash
cd desktop
pip install -r ../requirements.txt pyinstaller
npm run backend        # PyInstaller -> backend-dist/tripwire/
npm install
npm run dist           # electron-builder -> release/
```

## Run from source

```bash
cd desktop && npm install && npm start
```

In development the app runs `python3 ../tripwire_cli.py`. To use a different interpreter, set `TRIPWIRE_PYTHON`.
