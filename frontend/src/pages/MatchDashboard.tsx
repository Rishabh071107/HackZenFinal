import { useParams, Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { MatchesAPI, TeamsAPI } from "../api/client";
import StatCard from "../components/StatCard";
import TacticalGridPitch from "../components/TacticalGridPitch";
import TrackConsistencyMonitor from "../components/TrackConsistencyMonitor";

export default function MatchDashboard() {
  const { matchId } = useParams();
  const [videoError, setVideoError] = useState(false);

  const { data: match } = useQuery({ queryKey: ["match", matchId], queryFn: () => MatchesAPI.get(matchId!) });
  const { data: analytics } = useQuery({
    queryKey: ["match-analytics", matchId],
    queryFn: () => MatchesAPI.analytics(matchId!),
    retry: false,
  });
  const { data: heatmap } = useQuery({
    queryKey: ["match-heatmap", matchId],
    queryFn: () => MatchesAPI.heatmap(matchId!),
    retry: false,
  });
  const { data: teams } = useQuery({
    queryKey: ["match-teams", matchId],
    queryFn: () => MatchesAPI.teams(matchId!),
    retry: false,
  });

  const m = analytics?.metrics ?? {};

  const uniqueTeams = Array.from(new Map((teams || []).map((t) => [t.display_name || t.label, t])).values());

  return (
    <div className="max-w-6xl mx-auto px-6 py-12">
      <h1 className="text-3xl font-bold mb-1">{match?.name}</h1>
      <p className="text-slate-400 mb-8">
        {match?.team_a_name} vs {match?.team_b_name} · {match?.tournament}
      </p>

      <div className="grid md:grid-cols-2 gap-8 mb-10">
        <div className="card">
          <h2 className="font-semibold mb-4">Annotated Match Video</h2>
          {match?.status !== "completed" ? (
            <p className="text-sm text-slate-500">
              Video will be available once analysis completes (current status: {match?.status ?? "loading"}).
            </p>
          ) : videoError ? (
            <p className="text-sm text-red-400">
              Could not load the annotated video. Try refreshing the page or re-running analysis.
            </p>
          ) : match ? (
            <video
              key={match.id}
              controls
              playsInline
              preload="metadata"
              className="w-full rounded-lg bg-black"
              src={`/api/v1/files/video/${match.id}`}
              onError={() => setVideoError(true)}
            />
          ) : null}
        </div>
        <div className="card">
          <h2 className="font-semibold mb-4">Overall Match Occupancy Heatmap</h2>
          {heatmap ? (
            <img src={`/api/v1/files/heatmap-image/${heatmap.id}`} alt="Match heatmap" className="w-full rounded-lg" />
          ) : (
            <p className="text-sm text-slate-500">Not available yet.</p>
          )}
        </div>
      </div>

      <div className="card mb-8 border-accent-500/30 bg-pitch-900/60">
        <h2 className="font-semibold text-accent-400 mb-3 flex items-center gap-2">
          <span>🎯 VisionX OD-17 Tracking Performance Metrics</span>
        </h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <StatCard label="Confirmed Player Tracks" value={match?.confirmed_tracks ?? m.confirmed_tracks ?? "—"} />
          <StatCard label="ID Switches Prevented" value={match?.id_switches ?? m.id_switches ?? 0} />
          <StatCard label="Recovered Tracks (ReID)" value={match?.recovered_tracks ?? m.recovered_tracks ?? 0} />
          <StatCard label="Rejected Non-Players/Drones" value={match?.rejected_false_detections ?? m.rejected_false_detections ?? 0} />
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mb-10">
        <StatCard label="Players Detected" value={match?.confirmed_tracks ?? m.total_players_detected ?? "—"} />
        <StatCard label="Total Frames" value={m.total_frames ?? "—"} />
        <StatCard label="Duration (s)" value={m.match_duration_s ?? "—"} />
        <StatCard label="Avg Detection Conf." value={m.average_detection_confidence ?? "—"} />
        <StatCard label="Avg Tracking Acc." value={m.average_tracking_accuracy ?? "—"} />
      </div>

      {matchId && <TrackConsistencyMonitor matchId={matchId} />}

      <h2 className="text-xl font-bold mb-4">Teams</h2>
      <div className="grid md:grid-cols-2 gap-5 mb-10">
        {uniqueTeams.map((t) => (
          <Link to={`/teams/${t.id}`} key={t.id} className="card flex items-center gap-4 hover:border-accent-500/50 transition-colors">
            <span className="w-10 h-10 rounded-full" style={{ backgroundColor: t.jersey_color_hex }} />
            <div>
              <h3 className="font-semibold">{t.display_name || t.label}</h3>
              <p className="text-sm text-slate-400">View team analytics</p>
            </div>
          </Link>
        ))}
      </div>

      {uniqueTeams.length > 0 && (
        <div className="mb-10">
          <h2 className="text-xl font-bold mb-4">Tactical Zone Grid Analysis</h2>
          <div className="grid md:grid-cols-2 gap-6">
            {uniqueTeams.map((t) => (
              <TeamGridSection key={t.id} team={t} />
            ))}
          </div>
        </div>
      )}

      {match && (
        <div className="flex gap-3 flex-wrap">
          <button
            onClick={() => {
              MatchesAPI.analyze(match.id).then(() => {
                alert("Started frame-by-frame persistent re-tracking! Refresh page in a moment to view the updated video.");
              });
            }}
            className="text-sm bg-accent-600 hover:bg-accent-500 text-white font-medium px-4 py-2 rounded-lg transition-colors flex items-center gap-1.5"
          >
            ⚡ Re-Track & Re-Annotate Video
          </button>
          <a href={`/api/v1/matches/${match.id}/export/pdf`} className="text-sm bg-pitch-800 hover:bg-pitch-700 px-4 py-2 rounded-lg">Export PDF</a>
          <a href={`/api/v1/matches/${match.id}/export/json`} className="text-sm bg-pitch-800 hover:bg-pitch-700 px-4 py-2 rounded-lg">Export JSON</a>
          <a href={`/api/v1/matches/${match.id}/export/csv`} className="text-sm bg-pitch-800 hover:bg-pitch-700 px-4 py-2 rounded-lg">Export CSV</a>
        </div>
      )}
    </div>
  );
}

function TeamGridSection({ team }: { team: any }) {
  const { data: teamAnalytics } = useQuery({
    queryKey: ["team-analytics", team.id],
    queryFn: () => TeamsAPI.analytics(team.id),
    retry: false,
  });

  const zonePercentages = teamAnalytics?.metrics?.zone_percentages ?? {};
  return (
    <TacticalGridPitch
      zonePercentages={zonePercentages}
      teamName={team.display_name || team.label}
      jerseyColor={team.jersey_color_hex}
    />
  );
}
