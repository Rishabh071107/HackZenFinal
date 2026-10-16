import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { MatchesAPI, ConsistencyLogEvent } from "../api/client";
import { Activity, ShieldCheck, RefreshCw, AlertTriangle, Filter } from "lucide-react";

interface Props {
  matchId: string;
}

export default function TrackConsistencyMonitor({ matchId }: Props) {
  const [filterType, setFilterType] = useState<string>("ALL");
  const [searchTerm, setSearchTerm] = useState<string>("");

  const { data: logData, isLoading } = useQuery({
    queryKey: ["consistency-log", matchId],
    queryFn: () => MatchesAPI.consistencyLog(matchId),
    refetchInterval: 3000,
  });

  const events: ConsistencyLogEvent[] = logData?.events ?? [];

  const filteredEvents = events.filter((ev) => {
    if (filterType !== "ALL" && !ev.decision.includes(filterType)) {
      return false;
    }
    if (searchTerm) {
      const idMatch = String(ev.previous_track_id).includes(searchTerm);
      const decMatch = ev.decision.toLowerCase().includes(searchTerm.toLowerCase());
      if (!idMatch && !decMatch) return false;
    }
    return true;
  });

  const getDecisionBadge = (decision: string) => {
    if (decision.includes("REID_RECOVERY")) {
      return (
        <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-semibold bg-cyan-950/80 text-cyan-300 border border-cyan-800/60">
          <RefreshCw size={11} className="animate-spin" />
          ReID Recovered
        </span>
      );
    }
    if (decision.includes("MATCH_CONFIRMED")) {
      return (
        <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-semibold bg-emerald-950/80 text-emerald-300 border border-emerald-800/60">
          <ShieldCheck size={11} />
          Matched Active
        </span>
      );
    }
    if (decision.includes("NEW_TRACK")) {
      return (
        <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-semibold bg-amber-950/80 text-amber-300 border border-amber-800/60">
          <Activity size={11} />
          New Track
        </span>
      );
    }
    return (
      <span className="px-2 py-0.5 rounded text-[11px] font-semibold bg-slate-800 text-slate-300">
        {decision}
      </span>
    );
  };

  return (
    <div className="card mb-10 border border-pitch-700 bg-pitch-950/70 p-6 rounded-xl shadow-xl">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-6">
        <div>
          <h2 className="text-xl font-bold text-white flex items-center gap-2">
            <Activity className="text-accent-400" size={22} />
            <span>Track Consistency & ID Switch Monitor</span>
          </h2>
          <p className="text-xs text-slate-400 mt-1">
            Real-time multi-cue Hungarian association audit log, ReID feature recovery events, and track lifecycle metrics.
          </p>
        </div>

        {/* Live Filter Controls */}
        <div className="flex items-center gap-2">
          <div className="relative">
            <input
              type="text"
              placeholder="Filter by ID..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="bg-pitch-900 border border-pitch-700 rounded-lg px-3 py-1.5 text-xs text-slate-200 placeholder-slate-500 w-32 focus:outline-none focus:border-accent-500"
            />
          </div>
          <div className="flex items-center bg-pitch-900 border border-pitch-700 rounded-lg p-0.5 text-xs">
            <button
              onClick={() => setFilterType("ALL")}
              className={`px-2.5 py-1 rounded transition-colors ${filterType === "ALL" ? "bg-accent-500 text-pitch-950 font-bold" : "text-slate-400 hover:text-white"}`}
            >
              All
            </button>
            <button
              onClick={() => setFilterType("REID_RECOVERY")}
              className={`px-2.5 py-1 rounded transition-colors ${filterType === "REID_RECOVERY" ? "bg-cyan-500 text-pitch-950 font-bold" : "text-slate-400 hover:text-white"}`}
            >
              Recoveries
            </button>
            <button
              onClick={() => setFilterType("MATCH_CONFIRMED")}
              className={`px-2.5 py-1 rounded transition-colors ${filterType === "MATCH_CONFIRMED" ? "bg-emerald-500 text-pitch-950 font-bold" : "text-slate-400 hover:text-white"}`}
            >
              Matches
            </button>
          </div>
        </div>
      </div>

      {/* Summary KPI Badges */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-6 font-mono">
        <div className="bg-pitch-900/90 border border-pitch-800 rounded-lg p-3">
          <div className="text-[11px] text-slate-400 uppercase">Confirmed Tracks</div>
          <div className="text-xl font-bold text-accent-400">{logData?.confirmed_tracks ?? 0}</div>
        </div>
        <div className="bg-pitch-900/90 border border-pitch-800 rounded-lg p-3">
          <div className="text-[11px] text-slate-400 uppercase">ID Switches Prevented</div>
          <div className="text-xl font-bold text-emerald-400">{logData?.id_switches ?? 0}</div>
        </div>
        <div className="bg-pitch-900/90 border border-pitch-800 rounded-lg p-3">
          <div className="text-[11px] text-slate-400 uppercase">ReID Recoveries</div>
          <div className="text-xl font-bold text-cyan-400">{logData?.recovered_tracks ?? 0}</div>
        </div>
        <div className="bg-pitch-900/90 border border-pitch-800 rounded-lg p-3">
          <div className="text-[11px] text-slate-400 uppercase">Rejected False Dets</div>
          <div className="text-xl font-bold text-amber-400">{logData?.rejected_false_detections ?? 0}</div>
        </div>
      </div>

      {/* Event Timeline Table */}
      <div className="overflow-x-auto rounded-lg border border-pitch-800 bg-pitch-900/40">
        <table className="w-full text-left text-xs">
          <thead className="bg-pitch-900 text-slate-400 uppercase font-mono text-[10px] tracking-wider border-b border-pitch-800">
            <tr>
              <th className="py-2.5 px-3">Frame</th>
              <th className="py-2.5 px-3">Track ID</th>
              <th className="py-2.5 px-3">Decision</th>
              <th className="py-2.5 px-3 text-right">IoU Sim</th>
              <th className="py-2.5 px-3 text-right">App. Sim (D3)</th>
              <th className="py-2.5 px-3 text-right">Motion Sim</th>
              <th className="py-2.5 px-3 text-right">Geom Sim</th>
              <th className="py-2.5 px-3 text-right">Fused Cost</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-pitch-800/60 font-mono">
            {isLoading ? (
              <tr>
                <td colSpan={8} className="py-8 text-center text-slate-500">
                  Loading tracking consistency log...
                </td>
              </tr>
            ) : filteredEvents.length === 0 ? (
              <tr>
                <td colSpan={8} className="py-8 text-center text-slate-500">
                  No tracking events recorded yet. Run analysis on a match video to generate telemetry.
                </td>
              </tr>
            ) : (
              [...filteredEvents].reverse().slice(0, 50).map((ev, idx) => (
                <tr key={idx} className="hover:bg-white/5 transition-colors">
                  <td className="py-2 px-3 text-slate-300">#{ev.frame_number}</td>
                  <td className="py-2 px-3 font-bold text-accent-300">Player #{ev.previous_track_id}</td>
                  <td className="py-2 px-3">{getDecisionBadge(ev.decision)}</td>
                  <td className="py-2 px-3 text-right text-slate-300">{ev.iou_score.toFixed(3)}</td>
                  <td className="py-2 px-3 text-right text-emerald-400 font-semibold">{ev.appearance_score.toFixed(3)}</td>
                  <td className="py-2 px-3 text-right text-slate-300">{ev.motion_score.toFixed(3)}</td>
                  <td className="py-2 px-3 text-right text-slate-300">{ev.geometry_score.toFixed(3)}</td>
                  <td className="py-2 px-3 text-right">
                    <span className={`font-semibold ${ev.final_cost <= 0.5 ? "text-emerald-400" : ev.final_cost <= 0.8 ? "text-amber-400" : "text-rose-400"}`}>
                      {ev.final_cost.toFixed(3)}
                    </span>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
      {filteredEvents.length > 50 && (
        <div className="text-right text-[11px] text-slate-500 mt-2">
          Showing 50 most recent of {filteredEvents.length} events
        </div>
      )}
    </div>
  );
}
