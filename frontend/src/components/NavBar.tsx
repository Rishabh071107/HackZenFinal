import { Link } from "react-router-dom";
import { Activity } from "lucide-react";

export default function NavBar() {
  return (
    <header className="sticky top-0 z-50 backdrop-blur bg-pitch-950/80 border-b border-pitch-700/50">
      <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
        <Link to="/" className="flex items-center gap-2 font-bold text-lg">
          <Activity className="text-accent-400" size={22} />
          <span>Football Analytics</span>
        </Link>
        <Link
          to="/upload"
          className="bg-accent-500 hover:bg-accent-400 transition-colors text-pitch-950 font-semibold px-4 py-2 rounded-lg text-sm"
        >
          Upload Match
        </Link>
      </div>
    </header>
  );
}
