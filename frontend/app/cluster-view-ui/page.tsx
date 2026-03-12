"use client";

import React, { useState, useEffect } from 'react';
import { 
  Check, Fish, LayoutGrid, ArrowLeft, Loader2, 
  Pencil, AlertCircle, ZoomIn, ZoomOut, RotateCcw, X, Send
} from 'lucide-react';
import { TransformWrapper, TransformComponent } from "react-zoom-pan-pinch";
// 1. Import Auth Context
import { useAuth } from '@/app/context/AuthContext'; 

// ... (Interfaces remain exactly the same)
interface ClusterSummary { id: string; count: number; preview_url: string; }
interface ClusterImage { id: number; url: string; }
interface ClusterDetail { cluster: string; total: number; images: ClusterImage[]; }

export default function ClusterViewUI() {
  // 2. Access user data
  const { user } = useAuth();
  const isAdmin = user?.type === 'admin';

  const [clusters, setClusters] = useState<ClusterSummary[]>([]);
  const [selectedCluster, setSelectedCluster] = useState<ClusterDetail | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editValue, setEditValue] = useState("");
  const [viewingImage, setViewingImage] = useState<string | null>(null);
  const [selectedIndices, setSelectedIndices] = useState<number[]>([]); 
  const [selectedClusterIds, setSelectedClusterIds] = useState<string[]>([]); 
  const [isMoveModalOpen, setIsMoveModalOpen] = useState(false);
  const [isCreatingNew, setIsCreatingNew] = useState(false);
  const [newClusterName, setNewClusterName] = useState("");

  const API_BASE = "http://127.0.0.1:8000";

  useEffect(() => { fetchClusters(); }, []);

  // ... (fetchClusters, fetchClusterDetail, handleRename remain same)
  const fetchClusters = async () => {
    setIsLoading(true);
    try {
      const res = await fetch(`${API_BASE}/clusters`);
      if (!res.ok) throw new Error("Failed to fetch clusters");
      const data = await res.json();
      setClusters(data);
    } catch (err) { setError("Could not connect to the clustering backend."); }
    finally { setIsLoading(false); }
  };

  const fetchClusterDetail = async (clusterId: string) => {
    setIsLoading(true);
    setSelectedIndices([]); 
    try {
      const res = await fetch(`${API_BASE}/clusters/${clusterId}`);
      if (!res.ok) throw new Error("Failed to fetch cluster details");
      const data = await res.json();
      setSelectedCluster(data);
    } catch (err) { setError("Failed to load images."); }
    finally { setIsLoading(false); }
  };

  const handleRename = async (oldId: string) => {
    if (!isAdmin) return; // Security Check
    if (!editValue.trim() || editValue === oldId) { setEditingId(null); return; }
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
    } catch (err) { setError("Failed to rename."); }
  };

  // --- SELECTION LOGIC (Restricted to Admin) ---
  const toggleSelect = (index: number) => {
    if (!isAdmin) return; 
    setSelectedIndices(prev => 
      prev.includes(index) ? prev.filter(i => i !== index) : [...prev, index]
    );
  };

  const toggleClusterSelection = (id: string) => {
    if (!isAdmin) return;
    setSelectedClusterIds(prev => 
      prev.includes(id) ? prev.filter(i => i !== id) : [...prev, id]
    );
  };

  const handleSelectAllImages = () => { if (isAdmin && selectedCluster) setSelectedIndices(selectedIndices.length === selectedCluster.images.length ? [] : selectedCluster.images.map((_, i) => i)); };
  const handleSelectAllClusters = () => { if (isAdmin) setSelectedClusterIds(selectedClusterIds.length === clusters.length ? [] : clusters.map(c => c.id)); };

  const handleMove = async (targetId: string) => {
    if (!isAdmin || !selectedCluster) return;
    const movingAllImages = selectedIndices.length === selectedCluster.images.length;
    try {
      const realIds = selectedIndices.map(index => selectedCluster.images[index].id);
      const res = await fetch(`${API_BASE}/clusters/move`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ source_id: selectedCluster.cluster, target_id: targetId, indices: realIds }),
      });
      if (!res.ok) throw new Error("Move failed");
      setIsMoveModalOpen(false);
      setSelectedIndices([]);
      await fetchClusters(); 
      if (movingAllImages) setSelectedCluster(null);
      else await fetchClusterDetail(selectedCluster.cluster);
    } catch (err) { setError("Failed to move."); }
  };

  const handleConfirmTraining = async () => {
    if (!isAdmin || selectedClusterIds.length === 0) return;
    if (!window.confirm(`Exporting clusters?`)) return;
    setIsLoading(true);
    try {
      const res = await fetch(`${API_BASE}/clusters/confirm`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(selectedClusterIds),
      });
      if (res.ok) { setSelectedClusterIds([]); await fetchClusters(); }
    } catch (err) { setError("Export failed."); }
    finally { setIsLoading(false); }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-black-950 to-slate-900 text-white font-sans">
      <header className="relative border-b border-blue-900/30 backdrop-blur-sm bg-slate-900/50 sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="p-2 bg-gradient-to-br from-blue-500 to-cyan-500 rounded-lg"><LayoutGrid className="w-6 h-6" /></div>
            <h1 className="text-xl font-bold bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">Species Cluster Explorer</h1>
          </div>
          <div className="flex items-center gap-3">
            {/* 3. Hide Export button from students */}
            {isAdmin && selectedClusterIds.length > 0 && !selectedCluster && (
              <button onClick={handleConfirmTraining} className="flex items-center gap-2 px-6 py-2 bg-emerald-600 hover:bg-emerald-500 text-white rounded-lg transition-all shadow-lg font-bold border border-emerald-400/30">
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
          <div className="flex items-center justify-between gap-3 p-4 bg-red-500/10 border border-red-500/30 rounded-lg mb-8 text-red-300">
            <div className="flex items-center gap-3"><AlertCircle className="w-5 h-5" /><p className="text-sm">{error}</p></div>
            <button onClick={() => setError(null)} className="p-1 hover:bg-red-500/20 rounded-md"><X className="w-5 h-5" /></button>
          </div>
        )}

        {isLoading && <div className="flex flex-col items-center justify-center py-20"><Loader2 className="w-12 h-12 text-blue-500 animate-spin mb-4" /></div>}

        {!isLoading && !selectedCluster && (
          <div className="space-y-6">
            {/* 4. Only show selection bar to Admin */}
            {isAdmin && selectedClusterIds.length > 0 && (
              <div className="flex items-center justify-between bg-emerald-600/20 border border-emerald-500/50 p-4 rounded-xl sticky top-24 z-30">
                <div className="flex items-center gap-6">
                  <span className="font-bold text-emerald-400">{selectedClusterIds.length} Clusters Selected</span>
                  <button onClick={handleSelectAllClusters} className="text-xs font-bold text-slate-400 hover:text-white">Select All</button>
                </div>
                <button onClick={() => setSelectedClusterIds([])}><X className="w-5 h-5 text-emerald-400" /></button>
              </div>
            )}

          {clusters.length === 0 ? (
                // EMPTY STATE UI
                <div className="flex flex-col items-center justify-center py-32 bg-slate-900/20 border-2 border-dashed border-slate-800 rounded-3xl">
                  <div className="p-4 bg-slate-800/50 rounded-full mb-4">
                    <Fish className="w-12 h-12 text-slate-500" />
                  </div>
                  <h3 className="text-xl font-bold text-slate-300">No clusters available</h3>
                  <p className="text-slate-500 mt-2 max-w-xs text-center">
                    It looks like the clustering process hasn't been run yet or there is no data to display.
                  </p>
                </div>
              ) : (
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
              {clusters.map((c) => {
                const isIdSelected = selectedClusterIds.includes(c.id);
                return (
                  <div 
                    key={c.id}
                    onContextMenu={(e) => { e.preventDefault(); isAdmin && toggleClusterSelection(c.id); }}
                    onClick={() => {
                        if(isAdmin && selectedClusterIds.length > 0) toggleClusterSelection(c.id);
                        else fetchClusterDetail(c.id);
                    }}
                    className={`group bg-slate-900/40 border rounded-2xl p-4 cursor-pointer transition-all ${isIdSelected ? 'border-emerald-500 ring-4 ring-emerald-500/20' : 'border-slate-800 hover:border-blue-500/50'}`}
                  >
                    <div className="relative aspect-video rounded-xl overflow-hidden mb-4 bg-slate-950">
                      <img src={`${API_BASE}${c.preview_url}`} className={`w-full h-full object-cover group-hover:scale-110 ${isIdSelected ? 'opacity-40' : ''}`} />
                      <div className="absolute top-2 right-2 px-2 py-1 bg-black/60 rounded text-xs text-blue-400">{c.count} Items</div>
                    </div>
                    <h3 className="font-semibold uppercase text-sm text-slate-200">{c.id.replace('_', ' ')}</h3>
                  </div>
                );
              })}
            </div>
            )}
          </div>
        )}

        {!isLoading && selectedCluster && (
          <div className="space-y-6">
            <div className="bg-slate-900/50 p-6 rounded-2xl border border-slate-800 flex items-center justify-between">
              <div className="flex items-center gap-3">
                {isAdmin && editingId === selectedCluster.cluster ? (
                    <div className="flex items-center gap-2">
                        <input autoFocus className="bg-slate-800 border border-blue-500 rounded px-2 py-1 text-2xl font-bold outline-none" value={editValue} onChange={(e) => setEditValue(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && handleRename(selectedCluster.cluster)} />
                    </div>
                ) : (
                    <><h2 className="text-3xl font-bold text-white uppercase">{selectedCluster.cluster.replace('_', ' ')}</h2>
                    {/* 5. Hide Rename Pencil from students */}
                    {isAdmin && <button onClick={() => { setEditingId(selectedCluster.cluster); setEditValue(selectedCluster.cluster); }} className="p-2 text-slate-500 hover:text-blue-400"><Pencil className="w-5 h-5" /></button>}
                    </>
                )}
              </div>
            </div>

            {/* 6. Hide Move Toolbar from students */}
            {isAdmin && selectedIndices.length > 0 && (
              <div className="flex items-center justify-between bg-blue-600/20 border border-blue-500/50 p-4 rounded-xl sticky top-24 z-30">
                <div className="flex items-center gap-6">
                  <span className="font-bold text-blue-400">{selectedIndices.length} Selected</span>
                  <button onClick={handleSelectAllImages} className="text-xs font-bold text-slate-400 hover:text-white">Select All</button>
                </div>
                <button onClick={() => setIsMoveModalOpen(true)} className="bg-blue-500 hover:bg-blue-600 px-6 py-2 rounded-lg font-bold">Move to Cluster</button>
              </div>
            )}

            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
              {selectedCluster.images.map((image, index) => {
                const isSelected = selectedIndices.includes(index);
                return (
                  <div key={index} 
                    onClick={() => {
                        if(isAdmin && selectedIndices.length > 0) toggleSelect(index);
                        else setViewingImage(`${API_BASE}${image.url}`);
                    }} 
                    onContextMenu={(e) => { e.preventDefault(); isAdmin && toggleSelect(index); }} 
                    className={`group relative aspect-square rounded-xl overflow-hidden border transition-all cursor-pointer ${isSelected ? 'border-blue-500 ring-4 ring-blue-500/20' : 'border-slate-800'}`}
                  >
                    <img src={`${API_BASE}${image.url}`} className={`w-full h-full object-cover ${isSelected ? 'opacity-40' : ''}`} />
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </main>

      {/* --- MODALS (Only accessible via Admin actions anyway) --- */}
      {isAdmin && isMoveModalOpen && (
        <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/90 p-6">
           {/* ... (Existing Move Modal Code) ... */}
           <div className="bg-slate-900 border border-slate-700 rounded-3xl w-full max-w-4xl max-h-[85vh] overflow-hidden flex flex-col">
            <div className="p-6 border-b border-slate-800 flex justify-between items-center"><h2 className="text-xl font-bold italic">Relocate {selectedIndices.length} Images</h2><button onClick={() => setIsMoveModalOpen(false)}><X /></button></div>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4 p-6 overflow-y-auto">
              {clusters.filter(c => c.id !== selectedCluster?.cluster).map(c => (
                <div key={c.id} onClick={() => handleMove(c.id)} className="bg-slate-800/50 p-3 rounded-xl border border-slate-700 hover:border-blue-500 cursor-pointer group">
                    <img src={`${API_BASE}${c.preview_url}`} className="w-full h-24 object-cover rounded-lg mb-2" />
                    <p className="text-sm font-bold uppercase truncate">{c.id.replace('_', ' ')}</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {viewingImage && (
        <div className="fixed inset-0 z-[100] bg-black/95 flex items-center justify-center">
          <button onClick={() => setViewingImage(null)} className="absolute top-6 right-6 z-[120] p-4 bg-white/10 rounded-full text-white"><X className="w-8 h-8" /></button>
          <TransformWrapper initialScale={1} centerOnInit={true}>
            <TransformComponent wrapperStyle={{ width: "100%", height: "100%" }} contentStyle={{ display: "flex", alignItems: "center", justifyContent: "center" }}>
                <img src={viewingImage} className="max-h-[85vh] max-w-[85vw] object-contain shadow-2xl rounded-lg" />
            </TransformComponent>
          </TransformWrapper>
        </div>
      )}
    </div>
  );
}