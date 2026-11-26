const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('tripwire', {
  onLog: (fn) => ipcRenderer.on('log', (_e, s) => fn(s)),
  onDone: (fn) => ipcRenderer.on('done', (_e, code) => fn(code)),
  stopJob: (name) => ipcRenderer.send('stop-job', name),
  readText: (p) => ipcRenderer.invoke('read-text', p),
  openHome: (sub) => ipcRenderer.send('open-home', sub),
});
