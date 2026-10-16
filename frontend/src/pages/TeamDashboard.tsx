import { useParams, Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { TeamsAPI } from "../api/client";
import StatCard from "../components/StatCard";
import { playerSquadNumber } from "../utils/playerDisplay";
import TacticalGridPitch from "../components/TacticalGridPitch";

export default function TeamDashboard() {
  const { teamId } = useParams();

  const { data: team } = useQuery({
    queryKey: ["team", teamId],
    queryFn: () => TeamsAPI.get(teamId!),
  });
  const { data: analytics } = useQuery({
    queryKey: ["team-analytics", teamId],
    queryFn: () => TeamsAPI.analytics(teamId!),
    retry: false,
  });
  const { data: heatmap } = useQuery({
    queryKey: ["team-heatmap", teamId],
    queryFn: () => TeamsAPI.heatmap(teamId!),
    retry: false,
  });
  const { data: players } = useQuery({
    queryKey: ["team-players", teamId],
    queryFn: () => TeamsAPI.players(teamId!),
  });

  const m = analytics?.metrics ?? {};

  return (
    <div className="max-w-6xl mx-auto px-6 py-12">
      <h1 className="text-3xl font-bold mb-8">
        {team ? `${team.display_name || team.label} Analytics` : "Team Analytics"}
      </h1>

      <div className="grid md:grid-cols-2 gap-8 mb-10">
        <div className="card">
          <h2 className="font-semibold mb-4">Team Heatmap</h2>
          {heatmap ? (
            <img src={`/api/v1/files/heatmap-image/${heatmap.id}`} alt="Team heatmap" className="w-full rounded-lg" />
          ) : (
            <p className="text-sm text-slate-500">Not available yet.</p>
          )}
        </div>
        <div className="grid grid-cols-2 gap-4 content-start">
          <StatCard label="Width (m)" value={m.team_width_m ?? "—"} />
          <StatCard label="Depth (m)" value={m.team_depth_m ?? "—"} />
          <StatCard label="Compactness (m)" value={m.compactness_m ?? "—"} />
          <StatCard label="Total Distance (m)" value={m.total_distance_covered_m ?? "—"} />
          <StatCard label="Attacking %" value={m.attacking_percentage ?? "—"} />
          <StatCard label="Defensive %" value={m.defensive_percentage ?? "—"} />
        </div>
      </div>

      {team && analytics && (
        <div className="mb-10">
          <h2 className="text-xl font-bold mb-4 font-semibold text-slate-200">Tactical Zone Grid Analysis</h2>
          <TacticalGridPitch
            zonePercentages={m.zone_percentages ?? {}}
            teamName={team.display_name || team.label}
            jerseyColor={team.jersey_color_hex}
          />
        </div>
      )}

      <h2 className="text-xl font-bold mb-4">Players</h2>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {[...(players ?? [])]
          .sort(
            (a, b) =>
              (a.jersey_number ?? 999) - (b.jersey_number ?? 999) ||
              a.tracking_id - b.tracking_id,
          )
          .map((p) => (
          <Link to={`/players/${p.id}`} key={p.id} className="card text-center hover:border-accent-500/50 transition-colors">
            <div className="w-14 h-14 mx-auto rounded-full bg-pitch-800 flex items-center justify-center font-bold mb-2">
              #{playerSquadNumber(p)}
            </div>
            <p className="text-sm text-slate-300">{p.estimated_role ?? "Unclassified"}</p>
          </Link>
        ))}
      </div>
    </div>
  );
}
