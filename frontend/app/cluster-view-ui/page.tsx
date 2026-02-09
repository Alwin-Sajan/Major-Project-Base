"use client";

import React, { useState, useEffect } from 'react';
import { 
  Check, Fish, LayoutGrid, ArrowLeft, Loader2, 
  Pencil, AlertCircle, ZoomIn, ZoomOut, RotateCcw, X, Send
} from 'lucide-react';
import { TransformWrapper, TransformComponent } from "react-zoom-pan-pinch";

interface ClusterSummary {
  id: string;
  count: number;
  preview_url: string;
}

interface ClusterImage {
  id: number;
  url: string;
}

interface ClusterDetail {
  cluster: string;
  total: number;
  images: ClusterImage[];
}

export default function ClusterViewUI() {
  const [clusters, setClusters] = useState<ClusterSummary[]>([]);
  const [selectedCluster, setSelectedCluster] = useState<ClusterDetail | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editValue, setEditValue] = useState("");
  
  // Viewer State
  const [viewingImage, setViewingImage] = useState<string | null>(null);
  
  // Selection States
  const [selectedIndices, setSelectedIndices] = useState<number[]>([]); // For images inside a cluster
  const [selectedClusterIds, setSelectedClusterIds] = useState<string[]>([]); // For clusters in the grid
  
  const [isMoveModalOpen, setIsMoveModalOpen] = useState(false);
  const [isCreatingNew, setIsCreatingNew] = useState(false);
  const [newClusterName, setNewClusterName] = useState("");

  const API_BASE = "http://127.0.0.1:8000";

  useEffect(() => { fetchClusters(); }, []);

  const fetchClusters = async () => {
    setIsLoading(true);
    try {
      const res = await fetch(`${API_BASE}/clusters`);
      if (!res.ok) throw new Error("Failed to fetch clusters");
      const data = await res.json();
      setClusters(data);
    } catch (err) {
      setError("Could not connect to the clustering backend.");
    } finally {
      setIsLoading(false);
    }
  };

  const fetchClusterDetail = async (clusterId: string) => {
    setIsLoading(true);
    setSelectedIndices([]); 
    try {
      const res = await fetch(`${API_BASE}/clusters/${clusterId}`);
      if (!res.ok) throw new Error("Failed to fetch cluster details");
      const data = await res.json();
      setSelectedCluster(data);
    } catch (err) {
      setError("Failed to load images for this cluster.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleRename = async (oldId: string) => {
    if (!editValue.trim() || editValue === oldId) {
      setEditingId(null);
      return;
    }
    try {
      const res = await fetch(`${API_BASE}/clusters/${oldId}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ new_name: editValue.trim() }),
      });
      if (!res.ok) throw new Error("Rename failed");
      const data = await res.json();
      if (selectedCluster) setSelectedCluster({ ...selectedCluster, cluster: data.new_id });
      await fetchClusters();
      setEditingId(null);
    } catch (err) {
      setError("Failed to rename cluster.");
    }
  };

  // --- SELECTION LOGIC ---
  const toggleSelect = (index: number) => {
    setSelectedIndices(prev => 
      prev.includes(index) ? prev.filter(i => i !== index) : [...prev, index]
    );
  };

  const toggleClusterSelection = (id: string) => {
    setSelectedClusterIds(prev => 
      prev.includes(id) ? prev.filter(i => i !== id) : [...prev, id]
    );
  };

  // Global toggle for images inside a cluster
  const handleSelectAllImages = () => {
    if (!selectedCluster) return;
    if (selectedIndices.length === selectedCluster.images.length) {
      setSelectedIndices([]); 
    } else {
      setSelectedIndices(selectedCluster.images.map((_, i) => i));
    }
  };

  // Global toggle for clusters in the grid
  const handleSelectAllClusters = () => {
    if (selectedClusterIds.length === clusters.length) {
      setSelectedClusterIds([]);
    } else {
      setSelectedClusterIds(clusters.map(c => c.id));
    }
  };

  const handleMove = async (targetId: string) => {
    if (!selectedCluster) return;
    const movingAllImages = selectedIndices.length === selectedCluster.images.length;
    try {
      const realIds = selectedIndices.map(index => selectedCluster.images[index].id);
      const res = await fetch(`${API_BASE}/clusters/move`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          source_id: selectedCluster.cluster,
          target_id: targetId,
          indices: realIds
        }),
      });
      if (!res.ok) throw new Error("Move failed");
      setIsMoveModalOpen(false);
      setSelectedIndices([]);
      await fetchClusters(); 
      if (movingAllImages) setSelectedCluster(null);
      else await fetchClusterDetail(selectedCluster.cluster);
    } catch (err) {
      setError("Failed to move images.");
    }
  };

  const handleConfirmTraining = async () => {
    if (selectedClusterIds.length === 0) return;
    if (!window.confirm(`Exporting ${selectedClusterIds.length} clusters for training. Proceed?`)) return;
    setIsLoading(true);
    try {
      const res = await fetch(`${API_BASE}/clusters/confirm`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(selectedClusterIds),
      });
      if (res.ok) {
        setSelectedClusterIds([]);
        await fetchClusters();
      } else throw new Error("Export failed");
    } catch (err) {
      setError("Failed to export clusters.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-black-950 to-slate-900 text-white font-sans">
      <header className="relative border-b border-blue-900/30 backdrop-blur-sm bg-slate-900/50 sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="p-2 bg-gradient-to-br from-blue-500 to-cyan-500 rounded-lg">
              <LayoutGrid className="w-6 h-6" />
            </div>
            <h1 className="text-xl font-bold bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">
              Species Cluster Explorer
            </h1>
          </div>
          <div className="flex items-center gap-3">
            {selectedClusterIds.length > 0 && !selectedCluster && (
              <button onClick={handleConfirmTraining} className="flex items-center gap-2 px-6 py-2 bg-emerald-600 hover:bg-emerald-500 text-white rounded-lg transition-all shadow-lg font-bold border border-emerald-400/30 animate-in slide-in-from-right-4">
                <Send className="w-4 h-4" /> Confirm & Export
              </button>
            )}
            {selectedCluster && (
              <button onClick={() => setSelectedCluster(null)} className="flex items-center gap-2 px-4 py-2 bg-slate-800 hover:bg-slate-700 rounded-lg transition-colors border border-slate-700 text-sm">
                <ArrowLeft className="w-4 h-4" /> Back to Grid
              </button>
            )}
          </div>
        </div>
      </header>

      <main className="relative max-w-7xl mx-auto px-6 py-10">
{error && (
  <div className="flex items-center justify-between gap-3 p-4 bg-red-500/10 border border-red-500/30 rounded-lg mb-8 text-red-300 animate-in fade-in slide-in-from-top-2">
    <div className="flex items-center gap-3">
      <AlertCircle className="w-5 h-5 flex-shrink-0" />
      <p className="text-sm font-medium">{error}</p>
    </div>
    <button 
      onClick={() => setError(null)} 
      className="p-1 hover:bg-red-500/20 rounded-md transition-colors"
    >
      <X className="w-5 h-5" />
    </button>
  </div>
)}
        {isLoading && <div className="flex flex-col items-center justify-center py-20"><Loader2 className="w-12 h-12 text-blue-500 animate-spin mb-4" /><p className="text-slate-400 animate-pulse">Processing data...</p></div>}

        {/* VIEW 1: GRID OF ALL CLUSTERS */}
        {!isLoading && !selectedCluster && (
          <div className="space-y-6">
            {/* Cluster Selection Toolbar */}
            {selectedClusterIds.length > 0 && (
              <div className="flex items-center justify-between bg-emerald-600/20 border border-emerald-500/50 p-4 rounded-xl animate-in slide-in-from-top-4 backdrop-blur-md sticky top-24 z-30">
                <div className="flex items-center gap-6">
                  <span className="font-bold text-emerald-400 flex items-center gap-2">
                    <span className="w-2 h-2 bg-emerald-400 rounded-full animate-pulse"></span>
                    {selectedClusterIds.length} Clusters Selected
                  </span>
                  <button onClick={handleSelectAllClusters} className="text-xs uppercase tracking-widest font-bold text-slate-400 hover:text-white transition-colors">
                    {selectedClusterIds.length === clusters.length ? "Deselect All" : "Select All"}
                  </button>
                </div>
                <button onClick={() => setSelectedClusterIds([])} className="p-2 hover:bg-emerald-500/20 rounded-full transition-colors"><X className="w-5 h-5 text-emerald-400" /></button>
              </div>
            )}

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6 animate-in fade-in duration-500">
              {clusters.map((c) => {
                const isIdSelected = selectedClusterIds.includes(c.id);
                return (
                  <div 
                    key={c.id}
                    onContextMenu={(e) => { e.preventDefault(); toggleClusterSelection(c.id); }}
                    onClick={() => selectedClusterIds.length > 0 ? toggleClusterSelection(c.id) : fetchClusterDetail(c.id)}
                    className={`group bg-slate-900/40 border rounded-2xl p-4 cursor-pointer transition-all hover:scale-[1.02] relative ${
                      isIdSelected ? 'border-emerald-500 ring-4 ring-emerald-500/20 shadow-emerald-500/10' : 'border-slate-800 hover:border-blue-500/50'
                    }`}
                  >
                    {isIdSelected && <div className="absolute -top-2 -right-2 bg-emerald-500 rounded-full p-1.5 shadow-lg z-20"><Check className="w-4 h-4 text-white" /></div>}
                    <div className="relative aspect-video rounded-xl overflow-hidden mb-4 bg-slate-950">
                      <img src={`${API_BASE}${c.preview_url}`} className={`w-full h-full object-cover transition-transform group-hover:scale-110 ${isIdSelected ? 'opacity-40' : ''}`} />
                      <div className="absolute top-2 right-2 px-2 py-1 bg-black/60 backdrop-blur-md rounded text-xs font-bold text-blue-400 border border-blue-500/30">{c.count} Items</div>
                    </div>
                    <div className="flex items-center justify-between">
                      <h3 className={`font-semibold uppercase text-sm tracking-widest truncate mr-2 ${isIdSelected ? 'text-emerald-400' : 'text-slate-200'}`}>{c.id.replace('_', ' ')}</h3>
                      <Fish className={`w-4 h-4 ${isIdSelected ? 'text-emerald-500' : 'text-blue-500 opacity-50'}`} />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* VIEW 2: DETAIL VIEW */}
        {!isLoading && selectedCluster && (
          <div className="space-y-6 animate-in fade-in zoom-in-95 duration-500">
            <div className="bg-slate-900/50 p-6 rounded-2xl border border-slate-800 flex items-center justify-between">
              <div>
                <div className="flex items-center gap-3">
                  {editingId === selectedCluster.cluster ? (
                    <div className="flex items-center gap-2">
                      <input autoFocus className="bg-slate-800 border border-blue-500 rounded px-2 py-1 text-2xl font-bold outline-none" value={editValue} onChange={(e) => setEditValue(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && handleRename(selectedCluster.cluster)} />
                      <button onClick={() => handleRename(selectedCluster.cluster)} className="p-2 bg-blue-500 rounded-lg hover:bg-blue-600"><Check className="w-5 h-5 text-white" /></button>
                    </div>
                  ) : (
                    <><h2 className="text-3xl font-bold text-white uppercase">{selectedCluster.cluster.replace('_', ' ')}</h2><button onClick={() => { setEditingId(selectedCluster.cluster); setEditValue(selectedCluster.cluster); }} className="p-2 text-slate-500 hover:text-blue-400"><Pencil className="w-5 h-5" /></button></>
                  )}
                </div>
                <p className="text-slate-400 mt-1 text-sm">Cluster Management Mode • {selectedCluster.total} samples</p>
              </div>
            </div>

            {selectedIndices.length > 0 && (
              <div className="flex items-center justify-between bg-blue-600/20 border border-blue-500/50 p-4 rounded-xl sticky top-24 z-30 backdrop-blur-md animate-in slide-in-from-top-4">
                <div className="flex items-center gap-6">
                  <span className="font-bold text-blue-400 flex items-center gap-2"><span className="w-2 h-2 bg-blue-400 rounded-full animate-pulse"></span>{selectedIndices.length} Selected</span>
                  <button onClick={handleSelectAllImages} className="text-xs uppercase tracking-widest font-bold text-slate-400 hover:text-white transition-colors">
                    {selectedIndices.length === selectedCluster.images.length ? "Deselect All" : "Select All"}
                  </button>
                </div>
                <button onClick={() => setIsMoveModalOpen(true)} className="bg-blue-500 hover:bg-blue-600 px-6 py-2 rounded-lg font-bold flex items-center gap-2 shadow-lg active:scale-95 transition-all">Move to Cluster <ArrowLeft className="w-4 h-4 rotate-180" /></button>
              </div>
            )}

            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
              {selectedCluster.images.map((image, index) => {
                const isSelected = selectedIndices.includes(index);
                return (
                  <div key={index} onClick={() => selectedIndices.length > 0 ? toggleSelect(index) : setViewingImage(`${API_BASE}${image.url}`)} onContextMenu={(e) => { e.preventDefault(); toggleSelect(index); }} className={`group relative aspect-square rounded-xl overflow-hidden border transition-all duration-200 cursor-pointer ${isSelected ? 'border-blue-500 ring-4 ring-blue-500/20 z-10' : 'border-slate-800 hover:border-slate-600'}`}>
                    <img src={`${API_BASE}${image.url}`} className={`w-full h-full object-cover transition-all duration-300 ${isSelected ? 'opacity-40 scale-90' : 'group-hover:scale-110'}`} />
                    <div className="absolute inset-0 p-2 flex flex-col justify-between pointer-events-none">
                      <div className="flex justify-end">{isSelected && <div className="bg-blue-500 rounded-full p-1 shadow-lg animate-in zoom-in"><Check className="w-4 h-4 text-white" /></div>}</div>
                      <div className="self-start pointer-events-auto" onClick={(e) => e.stopPropagation()}><span className="text-[10px] text-slate-300 font-mono bg-black/60 backdrop-blur-sm px-1.5 py-1 rounded border border-white/10 opacity-0 group-hover:opacity-100 transition-opacity select-text cursor-text">{image.url.split('/').pop()}</span></div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </main>

      {/* --- MODALS (MOVE & VIEWER) --- */}
      {isMoveModalOpen && (
        <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/90 backdrop-blur-md p-6 animate-in fade-in">
          <div className="bg-slate-900 border border-slate-700 rounded-3xl w-full max-w-4xl max-h-[85vh] overflow-hidden flex flex-col shadow-2xl">
            <div className="p-6 border-b border-slate-800 flex justify-between items-center bg-slate-900 z-10"><h2 className="text-xl font-bold italic">Relocate {selectedIndices.length} Images</h2><button onClick={() => { setIsMoveModalOpen(false); setIsCreatingNew(false); }} className="p-2 hover:bg-slate-800 rounded-full"><X /></button></div>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4 p-6 overflow-y-auto bg-slate-950/30">
              {clusters.filter(c => c.id !== selectedCluster?.cluster).map(c => (
                <div key={c.id} onClick={() => handleMove(c.id)} className="bg-slate-800/50 p-3 rounded-xl border border-slate-700 hover:border-blue-500 cursor-pointer group"><img src={`${API_BASE}${c.preview_url}`} className="w-full h-24 object-cover rounded-lg mb-3 opacity-60 group-hover:opacity-100" /><p className="text-sm font-bold uppercase truncate">{c.id.replace('_', ' ')}</p><p className="text-xs text-slate-500">{c.count} images</p></div>
              ))}
              <div onClick={() => setIsCreatingNew(true)} className={`p-3 rounded-xl border-2 border-dashed flex flex-col items-center justify-center min-h-[160px] cursor-pointer ${isCreatingNew ? 'border-blue-500 bg-blue-500/10' : 'border-slate-700 hover:border-slate-500'}`}>
                {isCreatingNew ? <div className="w-full space-y-3" onClick={(e) => e.stopPropagation()}><input autoFocus className="w-full bg-slate-950 border border-blue-500/50 rounded-lg px-3 py-2 text-sm" value={newClusterName} onChange={(e) => setNewClusterName(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && handleMove(newClusterName)} /><div className="flex gap-2"><button onClick={() => handleMove(newClusterName)} className="flex-1 bg-blue-500 text-xs font-bold py-2 rounded-md">Create</button><button onClick={() => setIsCreatingNew(false)} className="px-3 bg-slate-800 text-xs py-2 rounded-md">Cancel</button></div></div> : <><LayoutGrid className="w-8 h-8 text-blue-400 mb-2" /><p className="text-sm font-bold">New Cluster</p></>}
              </div>
            </div>
          </div>
        </div>
      )}

      {viewingImage && (
        <div className="fixed inset-0 z-[100] bg-black/95 backdrop-blur-xl flex items-center justify-center animate-in fade-in">
          <button onClick={() => setViewingImage(null)} className="absolute top-6 right-6 z-[120] p-4 bg-white/10 hover:bg-white/20 rounded-full text-white"><X className="w-8 h-8" /></button>
          <TransformWrapper initialScale={1} centerOnInit={true}>
            {({ zoomIn, zoomOut, resetTransform }) => (
              <>
                <div className="absolute bottom-10 left-1/2 -translate-x-1/2 flex gap-6 p-4 bg-slate-900/90 border border-white/10 rounded-2xl z-[120] backdrop-blur-md"><button onClick={() => zoomIn()}><ZoomIn /></button><button onClick={() => zoomOut()}><ZoomOut /></button><button onClick={() => resetTransform()}><RotateCcw /></button></div>
                <TransformComponent wrapperStyle={{ width: "100%", height: "100%" }} contentStyle={{ display: "flex", alignItems: "center", justifyContent: "center" }}><img src={viewingImage} className="max-h-[85vh] max-w-[85vw] object-contain shadow-2xl rounded-lg" /></TransformComponent>
              </>
            )}
          </TransformWrapper>
        </div>
      )}
    </div>
  );
}