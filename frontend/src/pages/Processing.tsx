import { useParams, useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { useEffect } from "react";
import { MatchesAPI } from "../api/client";
import { CheckCircle2, Loader2 } from "lucide-react";

const STAGES = [
  "uploading", "extracting_frames", "detecting_players", "tracking_players",
  "classifying_teams", "mapping_pitch", "generating_heatmaps",
  "generating_trajectories", "generating_statistics", "preparing_dashboard", "completed",
];

const LABELS: Record<string, string> = {
  uploading: "Uploading",
  extracting_frames: "Extracting Frames",
  detecting_players: "Detecting Players",
  tracking_players: "Tracking Players",
  classifying_teams: "Classifying Teams",
  mapping_pitch: "Mapping Pitch Coordinates",
  generating_heatmaps: "Generating Heatmaps",
  generating_trajectories: "Generating Trajectories",
  generating_statistics: "Generating Statistics",
  preparing_dashboard: "Preparing Dashboard",
  completed: "Completed",
};

export default function Processing() {
  const { matchId } = useParams();
  const navigate = useNavigate();

  const { data: job } = useQuery({
    queryKey: ["status", matchId],
    queryFn: () => MatchesAPI.status(matchId!),
    refetchInterval: 2000,
    enabled: !!matchId,
  });

  useEffect(() => {
    if (job?.current_stage === "completed") {
      const t = setTimeout(() => navigate(`/matches/${matchId}`), 800);
      return () => clearTimeout(t);
    }
  }, [job, matchId, navigate]);

  const currentIndex = STAGES.indexOf(job?.current_stage ?? "uploading");

  return (
    <div className="max-w-xl mx-auto px-6 py-20">
      <h1 className="text-2xl font-bold mb-2">Analyzing Match</h1>
      <p className="text-slate-400 mb-8 text-sm">
        Running the full computer-vision pipeline. This can take a while for longer clips.
      </p>

      <div className="card space-y-4">
        <div className="w-full bg-pitch-800 rounded-full h-2 overflow-hidden">
          <div className="bg-accent-500 h-2 transition-all"
               style={{ width: `${job?.progress_percent ?? 0}%` }} />
        </div>

        <ul className="space-y-3 mt-4">
          {STAGES.filter((s) => s !== "completed").map((stage, i) => (
            <li key={stage} className="flex items-center gap-3 text-sm">
              {i < currentIndex ? (
                <CheckCircle2 className="text-accent-400" size={18} />
              ) : i === currentIndex ? (
                <Loader2 className="text-accent-400 animate-spin" size={18} />
              ) : (
                <div className="w-[18px] h-[18px] rounded-full border border-pitch-700" />
              )}
              <span className={i <= currentIndex ? "text-slate-100" : "text-slate-500"}>
                {LABELS[stage]}
              </span>
            </li>
          ))}
        </ul>

        {job?.error_message && (
          <p className="text-red-400 text-sm mt-4">Pipeline failed: {job.error_message}</p>
        )}
      </div>
    </div>
  );
}
