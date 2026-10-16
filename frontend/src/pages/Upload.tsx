import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { MatchesAPI } from "../api/client";
import { UploadCloud } from "lucide-react";

export default function Upload() {
  const navigate = useNavigate();
  const [file, setFile] = useState<File | null>(null);
  const [name, setName] = useState("");
  const [teamAName, setTeamAName] = useState("");
  const [teamBName, setTeamBName] = useState("");
  const [tournament, setTournament] = useState("");
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!file || !name || !teamAName || !teamBName) {
      setError("Match name, Team A name, Team B name and video file are required.");
      return;
    }
    setError(null);
    setUploading(true);

    const form = new FormData();
    form.append("name", name);
    form.append("team_a_name", teamAName);
    form.append("team_b_name", teamBName);
    form.append("tournament", tournament);
    form.append("file", file);

    try {
      // Simulated progress while the real upload streams in the background
      const interval = setInterval(() => setProgress((p) => Math.min(p + 8, 90)), 300);
      const match = await MatchesAPI.upload(form);
      clearInterval(interval);
      setProgress(100);
      await MatchesAPI.analyze(match.id);
      navigate(`/matches/${match.id}/processing`);
    } catch (err: any) {
      setError(err?.response?.data?.detail ?? "Upload failed.");
      setUploading(false);
    }
  }

  return (
    <div className="max-w-2xl mx-auto px-6 py-16">
      <h1 className="text-3xl font-bold mb-8">Upload Match Footage</h1>
      <form onSubmit={handleSubmit} className="card space-y-5">
        <div>
          <label className="stat-label block mb-1">Match Name</label>
          <input value={name} onChange={(e) => setName(e.target.value)} required
                 className="w-full bg-pitch-800 border border-pitch-700 rounded-lg px-3 py-2" />
        </div>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="stat-label block mb-1">Team A Name</label>
            <input value={teamAName} onChange={(e) => setTeamAName(e.target.value)} required placeholder="e.g. Arsenal"
                   className="w-full bg-pitch-800 border border-pitch-700 rounded-lg px-3 py-2" />
          </div>
          <div>
            <label className="stat-label block mb-1">Team B Name</label>
            <input value={teamBName} onChange={(e) => setTeamBName(e.target.value)} required placeholder="e.g. Chelsea"
                   className="w-full bg-pitch-800 border border-pitch-700 rounded-lg px-3 py-2" />
          </div>
        </div>
        <div>
          <label className="stat-label block mb-1">Tournament</label>
          <input value={tournament} onChange={(e) => setTournament(e.target.value)} placeholder="e.g. Premier League"
                 className="w-full bg-pitch-800 border border-pitch-700 rounded-lg px-3 py-2" />
        </div>

        <label className="border-2 border-dashed border-pitch-700 rounded-xl flex flex-col items-center justify-center py-10 cursor-pointer hover:border-accent-500/50 transition-colors">
          <UploadCloud className="text-accent-400 mb-2" size={32} />
          <span className="text-sm text-slate-400">
            {file ? file.name : "MP4, MOV or AVI — click to browse"}
          </span>
          <input
            type="file" accept=".mp4,.mov,.avi" className="hidden"
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          />
        </label>

        {uploading && (
          <div className="w-full bg-pitch-800 rounded-full h-2 overflow-hidden">
            <div className="bg-accent-500 h-2 transition-all" style={{ width: `${progress}%` }} />
          </div>
        )}

        {error && <p className="text-red-400 text-sm">{error}</p>}

        <button
          type="submit" disabled={uploading}
          className="w-full bg-accent-500 hover:bg-accent-400 disabled:opacity-50 transition-colors text-pitch-950 font-bold py-3 rounded-lg"
        >
          {uploading ? "Uploading…" : "Start Analysis"}
        </button>
      </form>
    </div>
  );
}
