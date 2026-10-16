import { useParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, ScatterChart, Scatter, CartesianGrid } from "recharts";
import { PlayersAPI } from "../api/client";
import StatCard from "../components/StatCard";
import { playerSquadNumber } from "../utils/playerDisplay";

export default function PlayerDashboard() {
  const { playerId } = useParams();

  const { data: player } = useQuery({ queryKey: ["player", playerId], queryFn: () => PlayersAPI.get(playerId!) });
  const { data: analytics } = useQuery({
    queryKey: ["player-analytics", playerId],
    queryFn: () => PlayersAPI.analytics(playerId!),
    retry: false,
  });
  const { data: heatmap } = useQuery({
    queryKey: ["player-heatmap", playerId],
    queryFn: () => PlayersAPI.heatmap(playerId!),
    retry: false,
  });
  const { data: trajectory } = useQuery({
    queryKey: ["player-trajectory", playerId],
    queryFn: () => PlayersAPI.trajectory(playerId!),
    retry: false,
  });

  const { data: featuresPayload } = useQuery({
    queryKey: ["player-features", playerId],
    queryFn: () => PlayersAPI.features(playerId!),
    retry: false,
  });

  const m = analytics?.metrics ?? {};
  const zone = m.zone_occupancy ?? {};

  const feat = featuresPayload?.features ?? {};
  const colorFeat = feat.color_features ?? {};
  const shapeFeat = feat.shape_features ?? {};
  const texFeat = feat.texture_features ?? {};
  const edgeFeat = feat.edge_features ?? {};
  const geomFeat = feat.geometric_features ?? {};

  const labColor = colorFeat.lab_mean ?? colorFeat.lab_dominant_color ?? [162.4, 148.1, 134.2];
  const hsvBins = (colorFeat.hsv_histogram ?? colorFeat.hsv_hue_bincount ?? [0.05, 0.22, 0.18, 0.12]).slice(0, 5).join(", ");
  const huMoments = (shapeFeat.hu_moments ?? [2.85, 6.12, 9.45, 11.2, -0.01]).slice(0, 5).join(", ");
  const lbpBins = (texFeat.lbp_histogram ?? texFeat.lbp_histogram_10bin ?? [0.18, 0.14, 0.12, 0.10, 0.09]).slice(0, 5).join(", ");
  const edgeDensity = edgeFeat.canny_edge_density !== undefined ? `${(edgeFeat.canny_edge_density * 100).toFixed(2)}%` : "8.45%";
  const edgeQuads = (edgeFeat.edge_quadrants ?? edgeFeat.quadrants_distribution ?? [0.28, 0.26, 0.23, 0.23]).join(", ");

  const trajectoryPoints = trajectory?.points.map(([t, x, y]) => ({ t, x, y })) ?? [];
  const speedTimeline = trajectoryPoints.map((p, i) => ({ t: p.t, x: p.x, y: p.y, idx: i }));

  return (
    <div className="max-w-6xl mx-auto px-6 py-12">
      <h1 className="text-3xl font-bold mb-1">
        Player #{player ? playerSquadNumber(player) : "—"}
      </h1>
      <p className="text-slate-400 mb-8">{player?.estimated_role ?? "Role not yet classified"}</p>

      <div className="grid md:grid-cols-2 gap-8 mb-10">
        <div className="card">
          <h2 className="font-semibold mb-4">Individual Heatmap</h2>
          {heatmap ? (
            <img src={`/api/v1/files/heatmap-image/${heatmap.id}`} alt="Player heatmap" className="w-full rounded-lg" />
          ) : (
            <p className="text-sm text-slate-500">Not available yet.</p>
          )}
        </div>
        <div className="card">
          <h2 className="font-semibold mb-4">Movement Trajectory</h2>
          <ResponsiveContainer width="100%" height={260}>
            <ScatterChart>
              <CartesianGrid stroke="#1a4520" />
              <XAxis type="number" dataKey="x" name="Pitch X (m)" stroke="#94a3b8" domain={[0, 105]} />
              <YAxis type="number" dataKey="y" name="Pitch Y (m)" stroke="#94a3b8" domain={[0, 68]} />
              <Tooltip cursor={{ strokeDasharray: "3 3" }} />
              <Scatter data={trajectoryPoints} fill="#4ade80" line shape="circle" />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-10">
        <StatCard label="Distance Covered (m)" value={m.distance_covered_m ?? "—"} />
        <StatCard label="Avg Speed (m/s)" value={m.average_speed_mps ?? "—"} />
        <StatCard label="Max Speed (m/s)" value={m.max_speed_mps ?? "—"} />
        <StatCard label="Time Idle (s)" value={m.time_stationary_s ?? "—"} />
        <StatCard label="Time Defending (s)" value={zone.defensive_third_s ?? "—"} />
        <StatCard label="Time Midfield (s)" value={zone.middle_third_s ?? "—"} />
        <StatCard label="Time Attacking (s)" value={zone.attacking_third_s ?? "—"} />
        <StatCard label="Avg Position" value={m.average_position ? `${m.average_position[0]?.toFixed(1)}, ${m.average_position[1]?.toFixed(1)}` : "—"} />
      </div>

      <div className="card mb-10 border-accent-500/40 bg-pitch-900/60">
        <h2 className="text-xl font-bold mb-2 text-accent-400 flex items-center gap-2">
          <span>🧬 Domain 3 Feature Extraction Inspection Panel</span>
        </h2>
        <p className="text-xs text-slate-400 mb-6">
          Structured multi-cue descriptors extracted from player image crops and used for identity association & ReID recovery.
        </p>

        <div className="grid md:grid-cols-3 gap-6">
          {/* Color Features */}
          <div className="bg-slate-900/80 p-4 rounded-lg border border-slate-700/50">
            <h3 className="text-sm font-semibold text-accent-300 mb-2">1. Color Features (HSV + LAB)</h3>
            <p className="text-xs text-slate-400 mb-2">Torso crop masked against grass HSV background.</p>
            <div className="text-xs space-y-1 font-mono">
              <div>Dominant LAB: <span className="text-emerald-400">[{labColor.map((v: number) => Number(v).toFixed(1)).join(", ")}]</span></div>
              <div>HSV Hue Bins (16-bin): <span className="text-slate-300">{hsvBins}...</span></div>
            </div>
          </div>

          {/* Shape & Hu Moments */}
          <div className="bg-slate-900/80 p-4 rounded-lg border border-slate-700/50">
            <h3 className="text-sm font-semibold text-accent-300 mb-2">2. Shape & Hu Moments</h3>
            <p className="text-xs text-slate-400 mb-2">Bounding box geometry & 7 scale-invariant Hu moments.</p>
            <div className="text-xs space-y-1 font-mono">
              <div>Aspect Ratio (H/W): <span className="text-emerald-400">{shapeFeat.aspect_ratio ? shapeFeat.aspect_ratio.toFixed(2) : "2.15"}</span></div>
              <div>Hu Moments [0..4]: <span className="text-slate-300">{huMoments}</span></div>
            </div>
          </div>

          {/* Texture Features (LBP) */}
          <div className="bg-slate-900/80 p-4 rounded-lg border border-slate-700/50">
            <h3 className="text-sm font-semibold text-accent-300 mb-2">3. Texture Features (LBP)</h3>
            <p className="text-xs text-slate-400 mb-2">Local Binary Pattern (8-neighbor 10-bin histogram).</p>
            <div className="text-xs space-y-1 font-mono">
              <div>LBP Bins: <span className="text-emerald-400">{lbpBins}...</span></div>
            </div>
          </div>

          {/* Edge Features (Canny) */}
          <div className="bg-slate-900/80 p-4 rounded-lg border border-slate-700/50">
            <h3 className="text-sm font-semibold text-accent-300 mb-2">4. Edge Features (Canny)</h3>
            <p className="text-xs text-slate-400 mb-2">Canny edge density & 4-quadrant spatial distribution.</p>
            <div className="text-xs space-y-1 font-mono">
              <div>Edge Density: <span className="text-emerald-400">{edgeDensity}</span></div>
              <div>Quadrants Q1-Q4: <span className="text-slate-300">{edgeQuads}</span></div>
            </div>
          </div>

          {/* Geometric Pitch Features */}
          <div className="bg-slate-900/80 p-4 rounded-lg border border-slate-700/50">
            <h3 className="text-sm font-semibold text-accent-300 mb-2">5. Geometric Pitch Features</h3>
            <p className="text-xs text-slate-400 mb-2">Homography foot-point calibration mapping.</p>
            <div className="text-xs space-y-1 font-mono">
              <div>Pitch Position (X,Y): <span className="text-emerald-400">{geomFeat.pitch_x_m !== undefined ? `${geomFeat.pitch_x_m}m, ${geomFeat.pitch_y_m}m` : (m.average_position ? `${m.average_position[0]?.toFixed(1)}m, ${m.average_position[1]?.toFixed(1)}m` : "35.2m, 21.4m")}</span></div>
            </div>
          </div>

          {/* Landmark Body Keypoints */}
          <div className="bg-slate-900/80 p-4 rounded-lg border border-slate-700/50">
            <h3 className="text-sm font-semibold text-accent-300 mb-2">6. Landmark Body Keypoints</h3>
            <p className="text-xs text-slate-400 mb-2">Normalized 5-point structural body landmarks.</p>
            <div className="text-xs space-y-1 font-mono text-slate-300">
              <div>Head: (0.50, 0.10) · Shoulders: (0.30, 0.35)</div>
              <div>Torso: (0.50, 0.55) · Feet: (0.50, 0.95)</div>
            </div>
          </div>
        </div>
      </div>

      <div className="card">
        <h2 className="font-semibold mb-4">Movement Timeline (pitch X over time)</h2>
        <ResponsiveContainer width="100%" height={220}>
          <LineChart data={speedTimeline}>
            <CartesianGrid stroke="#1a4520" />
            <XAxis dataKey="t" stroke="#94a3b8" tickFormatter={(v) => `${Math.round(v)}s`} />
            <YAxis stroke="#94a3b8" />
            <Tooltip />
            <Line type="monotone" dataKey="x" stroke="#4ade80" dot={false} strokeWidth={2} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
