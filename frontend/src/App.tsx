import { Routes, Route } from "react-router-dom";
import Landing from "./pages/Landing";
import Upload from "./pages/Upload";
import Processing from "./pages/Processing";
import MatchDashboard from "./pages/MatchDashboard";
import TeamDashboard from "./pages/TeamDashboard";
import PlayerDashboard from "./pages/PlayerDashboard";
import NavBar from "./components/NavBar";

export default function App() {
  return (
    <div className="min-h-screen">
      <NavBar />
      <Routes>
        <Route path="/" element={<Landing />} />
        <Route path="/upload" element={<Upload />} />
        <Route path="/matches/:matchId/processing" element={<Processing />} />
        <Route path="/matches/:matchId" element={<MatchDashboard />} />
        <Route path="/teams/:teamId" element={<TeamDashboard />} />
        <Route path="/players/:playerId" element={<PlayerDashboard />} />
      </Routes>
    </div>
  );
}
