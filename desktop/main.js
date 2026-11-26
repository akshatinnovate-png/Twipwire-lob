// TRIPWIRE-LOB desktop app.
// The Python backend (tripwire CLI, bundled with PyInstaller) serves the cockpit on a free local
// port; this window loads it. Menus start the live paper desk, reports, downloads and benchmarks
// as background jobs with their own log windows. Everything runs on this machine; paper trading
// only, no order is ever sent to an exchange.

const { app, BrowserWindow, Menu, dialog, shell, ipcMain } = require('electron');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

const isPackaged = app.isPackaged;
const HOME = path.join(app.getPath('documents'), 'TripwireLOB');
fs.mkdirSync(HOME, { recursive: true });

let win = null;
let cockpitUrl = null;
let server = null;
const jobs = new Map();   // name -> { proc, win, lines }

function backendCommand(args) {
  // packaged: resources/backend/tripwire(.exe); dev: python3 ../tripwire_cli.py
  if (isPackaged) {
    const exe = path.join(process.resourcesPath, 'backend', process.platform === 'win32' ? 'tripwire.exe' : 'tripwire');
    return [exe, ['--home', HOME, ...args]];
  }
  const py = process.env.TRIPWIRE_PYTHON || (process.platform === 'win32' ? 'python' : 'python3');
  return [py, ['-u', path.join(__dirname, '..', 'tripwire_cli.py'), '--home', HOME, ...args]];
}

function spawnBackend(args) {
  const [cmd, argv] = backendCommand(args);
  return spawn(cmd, argv, { env: { ...process.env, PYTHONUNBUFFERED: '1', PYTHONIOENCODING: 'utf-8' }, windowsHide: true });
}

// ------------------------------------------------------------------ cockpit server
function startServer(source) {
  if (server) { server.removeAllListeners('exit'); server.kill(); server = null; }
  cockpitUrl = null;
  if (win) win.loadFile(path.join(__dirname, 'ui', 'loading.html'));
  server = spawnBackend(['serve', '--source', source || 'synthetic']);
  let buf = '';
  const onData = (d) => {
    buf += d.toString();
    const m = buf.match(/TRIPWIRE_READY (http:\/\/[^\s]+)/);
    if (m && !cockpitUrl) {
      cockpitUrl = m[1];
      if (win) win.loadURL(cockpitUrl + '#cockpit');
    }
    if (buf.length > 20000) buf = buf.slice(-5000);
  };
  server.stdout.on('data', onData);
  server.stderr.on('data', onData);
  server.on('error', (e) => showBackendError(String(e)));
  server.on('exit', (code) => {
    if (!cockpitUrl) showBackendError(buf.slice(-3000) || `backend exited with code ${code}`);
  });
}

function showBackendError(text) {
  if (!win) return;
  win.loadFile(path.join(__dirname, 'ui', 'loading.html'), { query: { error: text } });
}

function control(body) {
  if (!cockpitUrl) return;
  fetch(cockpitUrl + '/control', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }).catch(() => {});
}

// ------------------------------------------------------------------ background jobs
function runJob(name, args, opts = {}) {
  if (jobs.has(name) && jobs.get(name).proc.exitCode === null) {
    const j = jobs.get(name); if (j.win && !j.win.isDestroyed()) j.win.show();
    return;
  }
  const w = new BrowserWindow({ width: 860, height: 560, title: name, backgroundColor: '#f4f1f2',
    webPreferences: { preload: path.join(__dirname, 'preload.js') } });
  w.setMenuBarVisibility(false);
  w.loadFile(path.join(__dirname, 'ui', 'log.html'), { query: { name, long: opts.long ? '1' : '' } });
  const proc = spawnBackend(args);
  const job = { proc, win: w, lines: [] };
  jobs.set(name, job);
  const push = (d) => {
    const s = d.toString(); job.lines.push(s);
    if (!w.isDestroyed()) w.webContents.send('log', s);
  };
  w.webContents.on('did-finish-load', () => job.lines.forEach(s => w.webContents.send('log', s)));
  proc.stdout.on('data', push);
  proc.stderr.on('data', push);
  proc.on('error', (e) => push(`\n[failed to start: ${e}]\n`));
  proc.on('exit', (code) => {
    push(`\n[finished with code ${code}]\n`);
    if (!w.isDestroyed()) w.webContents.send('done', code);
    if (opts.onDone) opts.onDone(code);
  });
  w.on('closed', () => { if (!opts.long && proc.exitCode === null) proc.kill(); });
}

ipcMain.on('stop-job', (_e, name) => { const j = jobs.get(name); if (j && j.proc.exitCode === null) j.proc.kill(); });
ipcMain.handle('read-text', (_e, p) => { const r = path.resolve(p); return r.startsWith(path.resolve(HOME)) && fs.existsSync(r) ? fs.readFileSync(r, 'utf8') : ''; });
ipcMain.on('open-home', (_e, sub) => shell.openPath(path.join(HOME, sub || '')));

function openReport() {
  runJob('Track record report', ['report'], {
    onDone: (code) => {
      const md = path.join(HOME, 'results', 'paper', 'track_record.md');
      if (code === 0 && fs.existsSync(md)) {
        const w = new BrowserWindow({ width: 1100, height: 800, title: 'Track record', backgroundColor: '#f4f1f2',
          webPreferences: { preload: path.join(__dirname, 'preload.js') } });
        w.setMenuBarVisibility(false);
        w.loadFile(path.join(__dirname, 'ui', 'report.html'), { query: { md, dir: path.dirname(md) } });
      }
    }
  });
}

// ------------------------------------------------------------------ menu
function buildMenu() {
  const src = (s, label) => ({ label, click: () => { if (cockpitUrl) control({ action: 'source', value: s }); else startServer(s); } });
  const template = [
    ...(process.platform === 'darwin' ? [{ role: 'appMenu' }] : []),
    { label: 'Cockpit', submenu: [
      src('synthetic', 'Source: synthetic (shock lab)'),
      src('AAPL', 'Source: AAPL (real LOBSTER day)'),
      src('AMZN', 'Source: AMZN (real LOBSTER day)'),
      src('MSFT', 'Source: MSFT (real LOBSTER day)'),
      src('BTC', 'Source: BTC-USDT live'),
      { type: 'separator' },
      { label: 'Concept flip   (key 1)', click: () => control({ action: 'shock', kind: 'concept' }) },
      { label: 'Covariate shock   (key 2)', click: () => control({ action: 'shock', kind: 'covariate' }) },
      { label: 'Liquidity sweep   (key 3)', click: () => control({ action: 'shock', kind: 'coupled' }) },
      { label: 'Back to calm   (key 0)', click: () => control({ action: 'shock', kind: 'calm' }) },
      { type: 'separator' },
      { label: 'Restart engine', click: () => startServer('synthetic') },
      { role: 'reload' }, { role: 'togglefullscreen' }, { role: 'toggleDevTools' },
      ...(process.platform === 'darwin' ? [] : [{ type: 'separator' }, { role: 'quit' }]),
    ]},
    { label: 'Desk', submenu: [
      { label: 'Start live paper desk (BTC + ETH)', click: () => runJob('Live paper desk', ['paper'], { long: true }) },
      { label: 'Stop live paper desk', click: () => { const j = jobs.get('Live paper desk'); if (j) j.proc.kill(); } },
      { type: 'separator' },
      { label: 'Track record report', click: openReport },
      { label: 'Edit events calendar…', click: () => shell.openPath(path.join(HOME, 'data', 'events.csv')) },
    ]},
    { label: 'Data', submenu: [
      { label: 'Download LOBSTER samples (AAPL/AMZN/MSFT)', click: () => runJob('Download LOBSTER', ['fetch', 'lobster']) },
      { label: 'Build real-data caches', click: () => runJob('Build caches', ['cache']) },
      { label: 'Record BTC book (45 min, for warm starts)', click: () => runJob('Record BTC', ['record', '--minutes', '45'], { long: true }) },
      { label: 'Download crypto history (Tardis free days)', click: () => runJob('Download crypto', ['fetch', 'crypto', '--months', '3']) },
      { type: 'separator' },
      { label: 'Open data folder', click: () => shell.openPath(HOME) },
    ]},
    { label: 'Benchmarks', submenu: [
      { label: 'Desk benchmark (H1–H3, LOBSTER)', click: () => runJob('Desk benchmark', ['bench', 'desk']) },
      { label: 'Crypto desk benchmark (evidence B)', click: () => runJob('Crypto benchmark', ['bench', 'crypto']) },
      { label: 'Forecasting benchmark', click: () => runJob('Forecasting benchmark', ['bench', 'forecast']) },
      { type: 'separator' },
      { label: 'Stage preflight check', click: () => runJob('Preflight', ['preflight']) },
      { label: 'Open results folder', click: () => shell.openPath(path.join(HOME, 'results')) },
    ]},
    { label: 'Help', submenu: [
      { label: 'Project on GitHub', click: () => shell.openExternal('https://github.com/akshatinnovate-png/Twipwire-lob') },
      { label: 'About', click: () => dialog.showMessageBox(win, { type: 'info', title: 'TRIPWIRE-LOB',
        message: 'TRIPWIRE-LOB ' + app.getVersion(),
        detail: 'Model-risk guard for order-book trading models.\nPaper trading only: no order ever leaves this machine.\nData folder: ' + HOME }) },
    ]},
  ];
  Menu.setApplicationMenu(Menu.buildFromTemplate(template));
}

app.whenReady().then(() => {
  win = new BrowserWindow({ width: 1500, height: 980, title: 'TRIPWIRE-LOB', backgroundColor: '#f2879f',
    webPreferences: { preload: path.join(__dirname, 'preload.js') } });
  win.webContents.setWindowOpenHandler(({ url }) => { shell.openExternal(url); return { action: 'deny' }; });
  buildMenu();
  startServer('synthetic');
});

function shutdown() {
  if (server) { server.removeAllListeners('exit'); server.kill(); }
  for (const j of jobs.values()) if (j.proc.exitCode === null) j.proc.kill();
}
app.on('before-quit', (e) => {
  const desk = jobs.get('Live paper desk');
  if (desk && desk.proc.exitCode === null) {
    const r = dialog.showMessageBoxSync(win, { type: 'question', buttons: ['Stop it and quit', 'Cancel'], defaultId: 1,
      message: 'The live paper desk is still recording its track record.', detail: 'Quitting stops it. It resumes from its logs next time.' });
    if (r === 1) { e.preventDefault(); return; }
  }
  shutdown();
});
app.on('window-all-closed', () => app.quit());
