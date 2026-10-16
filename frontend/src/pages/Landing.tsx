import { motion } from "framer-motion";
import { Link } from "react-router-dom";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { MatchesAPI } from "../api/client";
import { Flame, Map, Users, TrendingUp, Trash2 } from "lucide-react";

const features = [
  { icon: Map, title: "Movement Heatmaps", desc: "KDE-based occupancy heatmaps for every player, team, and the full match." },
  { icon: Users, title: "Automatic Team ID", desc: "Jersey-color clustering (HSV/LAB + KMeans) splits players into teams with no manual tagging." },
  { icon: TrendingUp, title: "Tactical Analytics", desc: "Distance, speed, zone occupancy, compactness, and formation shape, computed automatically." },
  { icon: Flame, title: "Trajectory Replay", desc: "Smoothed player paths mapped onto real pitch coordinates via homography." },
];

export default function Landing() {
  const queryClient = useQueryClient();
  const { data: matches } = useQuery({ queryKey: ["matches"], queryFn: MatchesAPI.list });

  const deleteMutation = useMutation({
    mutationFn: MatchesAPI.delete,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["matches"] });
    },
  });

  return (
    <div className="max-w-7xl mx-auto px-6">
      <section className="py-24 text-center">
        <motion.h1
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="text-5xl md:text-6xl font-extrabold tracking-tight"
        >
          Turn match footage into
          <span className="text-accent-400"> tactical intelligence</span>
        </motion.h1>
        <motion.p
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.1 }}
          className="mt-6 text-lg text-slate-400 max-w-2xl mx-auto"
        >
          Upload a broadcast video. Get automatic player detection, tracking,
          team classification, heatmaps, trajectories and full tactical analytics.
        </motion.p>
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.2 }}
          className="mt-10"
        >
          <Link
            to="/upload"
            className="inline-block bg-accent-500 hover:bg-accent-400 transition-colors text-pitch-950 font-bold px-8 py-4 rounded-xl text-lg shadow-lg shadow-accent-500/20"
          >
            Upload Match
          </Link>
        </motion.div>
      </section>

      <section className="grid md:grid-cols-4 gap-5 pb-20">
        {features.map((f, i) => (
          <motion.div
            key={f.title}
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.4, delay: i * 0.08 }}
            className="card"
          >
            <f.icon className="text-accent-400 mb-3" size={28} />
            <h3 className="font-semibold mb-2">{f.title}</h3>
            <p className="text-sm text-slate-400">{f.desc}</p>
          </motion.div>
        ))}
      </section>

      {matches && matches.length > 0 && (
        <section className="pb-24">
          <h2 className="text-2xl font-bold mb-5">Recent Analyses</h2>
          <div className="grid md:grid-cols-3 gap-5">
            {matches.slice(0, 6).map((m) => (
              <div key={m.id} className="relative group">
                <Link to={`/matches/${m.id}`} className="card hover:border-accent-500/50 transition-colors block h-full pr-12">
                  <h3 className="font-semibold">{m.name}</h3>
                  <p className="text-sm text-slate-400">{m.team_a_name} vs {m.team_b_name}</p>
                  <span className="inline-block mt-3 text-xs uppercase tracking-wide bg-pitch-800 px-2 py-1 rounded">
                    {m.status.replace(/_/g, " ")}
                  </span>
                </Link>
                <button
                  onClick={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    if (confirm("Are you sure you want to delete this analysis? This will remove all associated files and data.")) {
                      deleteMutation.mutate(m.id);
                    }
                  }}
                  className="absolute top-4 right-4 text-slate-400 hover:text-red-500 transition-colors p-1.5 rounded-lg hover:bg-pitch-800"
                  title="Delete analysis"
                >
                  <Trash2 size={18} />
                </button>
              </div>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}