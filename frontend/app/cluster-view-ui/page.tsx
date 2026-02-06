"use client";

import React, { useState, useEffect } from 'react';
import { Check, Fish, LayoutGrid, ArrowLeft, Image as ImageIcon, Loader2, Pencil, AlertCircle } from 'lucide-react';


interface ClusterSummary {
  id: string;
  count: number;
  preview_url: string;
}

interface ClusterDetail {
  cluster: string;
  total: number;
  images: string[];
}

export default function ClusterViewUI() {
  const [clusters, setClusters] = useState<ClusterSummary[]>([]);
  const [selectedCluster, setSelectedCluster] = useState<ClusterDetail | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editValue, setEditValue] = useState("");

  const API_BASE = "http://127.0.0.1:8000";

  // Fetch all clusters on load
  useEffect(() => {
    fetchClusters();
  }, []);

  const fetchClusters = async () => {
    setIsLoading(true);
    // setSelectedCluster(null); // <-- REMOVE OR COMMENT THIS LINE
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
      
      // 1. Update the Detail View state locally so we stay on this page
      if (selectedCluster) {
        setSelectedCluster({
          ...selectedCluster,
          cluster: data.new_id // Use the new name from backend response
        });
      }

      // 2. Silently update the background list without resetting selection
      const listRes = await fetch(`${API_BASE}/clusters`);
      if (listRes.ok) {
        const newList = await listRes.json();
        setClusters(newList);
      }

      setEditingId(null);
    } catch (err) {
      setError("Failed to rename cluster.");
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-black-950 to-slate-900 text-white">
      {/* Animated background elements */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-blue-500/10 rounded-full blur-3xl animate-pulse"></div>
        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-cyan-500/10 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '1s' }}></div>
      </div>

      {/* Header */}
      <header className="relative border-b border-blue-900/30 backdrop-blur-sm bg-slate-900/50">
        <div className="max-w-7xl mx-auto px-6 py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="p-3 bg-gradient-to-br from-blue-500 to-cyan-500 rounded-xl">
                <LayoutGrid className="w-8 h-8" />
              </div>
              <div>
                <h1 className="text-2xl font-bold bg-gradient-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">
                  Species Cluster Discovery <span className='text-red-500 font-medium'>Explorer</span>
                </h1>
                <p className="text-xs text-slate-400">DBSCAN + ConvNeXt Embedding Visualization</p>
              </div>
            </div>
              {selectedCluster && (
                <button 
                  onClick={() => setSelectedCluster(null)} // Explicitly set to null here
                  className="flex items-center gap-2 px-4 py-2 bg-slate-800 hover:bg-slate-700 rounded-lg transition-colors border border-slate-700"
                >
                  <ArrowLeft className="w-4 h-4" /> Back to Grid
                </button>
              )}
          </div>
        </div>
      </header>

      <main className="relative max-w-7xl mx-auto px-6 py-12">
        {/* Error State */}
        {error && (
          <div className="flex items-start gap-3 p-4 bg-red-500/10 border border-red-500/30 rounded-lg mb-8">
            <AlertCircle className="w-6 h-6 text-red-400 flex-shrink-0" />
            <p className="text-red-300">{error}</p>
          </div>
        )}

        {/* Loading State */}
        {isLoading && (
          <div className="flex flex-col items-center justify-center py-20">
            <Loader2 className="w-12 h-12 text-blue-500 animate-spin mb-4" />
            <p className="text-slate-400 animate-pulse">Scanning clusters...</p>
          </div>
        )}

        {/* Grid View: All Clusters */}
        {!isLoading && !selectedCluster && (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6 animate-in fade-in slide-in-from-bottom-4 duration-700">
            {clusters.map((c) => (
              <div 
                key={c.id}
                onClick={() => fetchClusterDetail(c.id)}
                className="group bg-slate-900/50 backdrop-blur-sm border border-slate-800 rounded-2xl p-4 cursor-pointer hover:border-blue-500/50 transition-all duration-300 hover:scale-[1.02]"
              >
                <div className="relative aspect-video rounded-xl overflow-hidden mb-4 border border-slate-800">
                  <img 
                    src={`${API_BASE}${c.preview_url}`} 
                    alt={c.id} 
                    className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-110"
                  />
                  <div className="absolute top-2 right-2 px-2 py-1 bg-black/60 backdrop-blur-md rounded-md text-xs font-bold text-blue-400 border border-blue-500/30">
                    {c.count} Images
                  </div>
                </div>
                <div className="flex items-center justify-between">
                  <h3 className="font-semibold text-slate-200 uppercase tracking-wider">{c.id.replace('_', ' ')}</h3>
                  <div className="p-2 bg-blue-500/10 rounded-full group-hover:bg-blue-500/20 transition-colors">
                    <Fish className="w-4 h-4 text-blue-400" />
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Detail View: Single Cluster Images */}
        {!isLoading && selectedCluster && (
          <div className="space-y-8 animate-in fade-in zoom-in-95 duration-500">
            <div className="bg-slate-900/50 p-6 rounded-2xl border border-slate-800 flex items-center justify-between">
              <div>
                <div className="flex items-center gap-3">
                  {editingId === selectedCluster.cluster ? (
                    <div className="flex items-center gap-2">
                      <input
                        autoFocus
                        className="bg-slate-800 border border-blue-500 rounded px-2 py-1 text-2xl font-bold outline-none"
                        value={editValue}
                        onChange={(e) => setEditValue(e.target.value)}
                        onKeyDown={(e) => e.key === 'Enter' && handleRename(selectedCluster.cluster)}
                      />
                      <button onClick={() => handleRename(selectedCluster.cluster)} className="p-2 bg-green-500/20 text-green-400 rounded-lg">
                        <Check className="w-6 h-6" />
                      </button>
                    </div>
                  ) : (
                    <>
                      <h2 className="text-3xl font-bold text-white uppercase">
                        {selectedCluster.cluster.replace('_', ' ')}
                      </h2>
                      <button 
                        onClick={() => {
                          setEditingId(selectedCluster.cluster);
                          setEditValue(selectedCluster.cluster);
                        }}
                        className="p-1.5 text-slate-500 hover:text-blue-400 transition-colors"
                      >
                        <Pencil className="w-5 h-5" />
                      </button>
                    </>
                  )}
                </div>
                <p className="text-slate-400">Showing {selectedCluster.total} images grouped in this density zone.</p>
              </div>
              <div className="hidden md:flex gap-4">
                <div className="text-center px-6 py-2 bg-slate-800/50 rounded-xl border border-slate-700">
                  <p className="text-xs text-slate-500">Confidence</p>
                  <p className="text-blue-400 font-bold italic">High</p>
                </div>
              </div>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-5 gap-4">
              {selectedCluster.images.map((url, index) => (
                <div key={index} className="group relative aspect-square rounded-xl overflow-hidden border border-slate-800 bg-slate-900 shadow-xl">
                  <img 
                    src={`${API_BASE}${url}`} 
                    alt={`Item ${index}`} 
                    className="w-full h-full object-cover transition-transform duration-300 group-hover:scale-110"
                  />
                  <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity flex items-end p-3">
                    <span className="text-[10px] text-slate-400 font-mono">IMG_IDX: {url.split('/').pop()}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}