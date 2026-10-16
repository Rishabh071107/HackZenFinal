import { useState } from "react";

interface TacticalGridPitchProps {
  zonePercentages: Record<string, number>;
  teamName: string;
  jerseyColor: string;
}

// Convert hex color to rgba string
const hexToRgba = (hex: string, alpha: number) => {
  const cleanHex = hex.replace("#", "");
  const r = parseInt(cleanHex.substring(0, 2), 16) || 0;
  const g = parseInt(cleanHex.substring(2, 4), 16) || 0;
  const b = parseInt(cleanHex.substring(4, 6), 16) || 0;
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
};

export default function TacticalGridPitch({ zonePercentages, teamName, jerseyColor }: TacticalGridPitchProps) {
  const [hoveredZone, setHoveredZone] = useState<number | null>(null);

  // Rows and Columns setup:
  // Col 1 to 6 (length 105m, 17.5m cols)
  // Row 1 (Left wing, 0-17m), Row 2 (Center, 17-51m), Row 3 (Right wing, 51-68m)
  const rows = [
    { label: "Left Wing", h: "25%" },
    { label: "Center Channel", h: "50%" },
    { label: "Right Wing", h: "25%" },
  ];
  const cols = Array.from({ length: 6 }, (_, i) => `Col ${i + 1}`);

  return (
    <div className="card bg-pitch-950/40 p-6 border border-pitch-800">
      <div className="flex justify-between items-center mb-4">
        <div>
          <h3 className="font-semibold text-lg flex items-center gap-2">
            <span className="w-3.5 h-3.5 rounded-full inline-block" style={{ backgroundColor: jerseyColor }} />
            {teamName} Grid Occupancy
          </h3>
          <p className="text-xs text-slate-400">18-Zone Tactical Position Analysis</p>
        </div>
        {hoveredZone !== null && (
          <div className="text-right">
            <span className="text-xs text-slate-400 font-mono">Zone {hoveredZone}</span>
            <p className="text-sm font-bold text-accent-400">
              {hoveredZone === 14 ? (
                <span>Zone 14 (Golden Zone) · {zonePercentages[String(hoveredZone)] ?? 0}%</span>
              ) : hoveredZone === 13 || hoveredZone === 15 ? (
                <span>Half-space Entry · {zonePercentages[String(hoveredZone)] ?? 0}%</span>
              ) : (
                <span>Occupancy · {zonePercentages[String(hoveredZone)] ?? 0}%</span>
              )}
            </p>
          </div>
        )}
      </div>

      {/* Pitch Outer Wrapper */}
      <div className="relative w-full aspect-[105/68] bg-pitch-900 border border-white/20 rounded-xl overflow-hidden shadow-2xl">
        
        {/* SVG Pitch Markings */}
        <svg className="absolute inset-0 w-full h-full pointer-events-none" viewBox="0 0 105 68" xmlns="http://www.w3.org/2000/svg">
          {/* Pitch background shading (striped effect) */}
          {Array.from({ length: 12 }).map((_, i) => (
            <rect
              key={i}
              x={i * 8.75}
              y={0}
              width={8.75}
              height={68}
              fill={i % 2 === 0 ? "rgba(255, 255, 255, 0.015)" : "transparent"}
            />
          ))}

          {/* Center line */}
          <line x1={52.5} y1={0} x2={52.5} y2={68} stroke="rgba(255, 255, 255, 0.15)" strokeWidth={0.5} />
          {/* Center circle */}
          <circle cx={52.5} cy={34} r={9.15} fill="none" stroke="rgba(255, 255, 255, 0.15)" strokeWidth={0.5} />
          {/* Center spot */}
          <circle cx={52.5} cy={34} r={0.3} fill="rgba(255, 255, 255, 0.3)" />

          {/* Left Penalty Box */}
          <rect x={0} y={13.84} width={16.5} height={40.32} fill="none" stroke="rgba(255, 255, 255, 0.15)" strokeWidth={0.5} />
          <rect x={0} y={24.84} width={5.5} height={18.32} fill="none" stroke="rgba(255, 255, 255, 0.15)" strokeWidth={0.5} />
          <path d="M 16.5,29.15 A 9.15,9.15 0 0,1 16.5,38.85" fill="none" stroke="rgba(255, 255, 255, 0.15)" strokeWidth={0.5} />

          {/* Right Penalty Box */}
          <rect x={88.5} y={13.84} width={16.5} height={40.32} fill="none" stroke="rgba(255, 255, 255, 0.15)" strokeWidth={0.5} />
          <rect x={99.5} y={24.84} width={5.5} height={18.32} fill="none" stroke="rgba(255, 255, 255, 0.15)" strokeWidth={0.5} />
          <path d="M 88.5,29.15 A 9.15,9.15 0 0,0 88.5,38.85" fill="none" stroke="rgba(255, 255, 255, 0.15)" strokeWidth={0.5} />
        </svg>

        {/* 18-Zone Interactive Grid Overlay */}
        <div className="absolute inset-0 grid grid-cols-6 h-full w-full">
          {cols.map((_, colIdx) => (
            <div key={colIdx} className="grid grid-rows-[25%_50%_25%] h-full">
              {rows.map((_, rowIdx) => {
                // Numbering layout: Col 1 is 1,2,3; Col 2 is 4,5,6 etc.
                const zoneNum = colIdx * 3 + rowIdx + 1;
                const pct = zonePercentages[String(zoneNum)] ?? 0.0;
                
                // Color scaling based on occupancy
                const cellOpacity = Math.min(pct / 20.0, 0.75); // fully shaded at 20%+
                const isZone14 = zoneNum === 14;
                const isHalfSpace = zoneNum === 13 || zoneNum === 15;

                return (
                  <div
                    key={rowIdx}
                    onMouseEnter={() => setHoveredZone(zoneNum)}
                    onMouseLeave={() => setHoveredZone(null)}
                    style={{
                      backgroundColor: hexToRgba(jerseyColor, cellOpacity),
                      transition: "all 200ms ease",
                    }}
                    className={`
                      relative flex flex-col items-center justify-center border border-white/5 cursor-pointer group
                      ${isZone14 ? "border-amber-400/40 shadow-[inset_0_0_8px_rgba(251,191,36,0.1)]" : ""}
                      ${isHalfSpace ? "border-sky-400/20" : ""}
                      hover:bg-white/5 hover:border-white/20
                    `}
                  >
                    {/* Zone Border Markers */}
                    {isZone14 && (
                      <div className="absolute inset-0 border-2 border-amber-400/50 rounded pointer-events-none z-10 animate-pulse" />
                    )}
                    {isHalfSpace && (
                      <div className="absolute inset-0 border border-dashed border-sky-400/40 rounded pointer-events-none z-10" />
                    )}

                    {/* Zone number & percentage text */}
                    <span className="text-[10px] font-mono text-slate-400/60 select-none">
                      {isZone14 ? "Zone 14" : isHalfSpace ? `HS ${zoneNum}` : zoneNum}
                    </span>
                    <span className="text-sm font-bold text-white tracking-tight">
                      {pct > 0 ? `${pct}%` : "0%"}
                    </span>

                    {/* Hover labels */}
                    {isZone14 && (
                      <span className="absolute bottom-1 text-[8px] uppercase tracking-wider text-amber-300 font-extrabold select-none opacity-0 group-hover:opacity-100 transition-opacity">
                        Golden Zone
                      </span>
                    )}
                  </div>
                );
              })}
            </div>
          ))}
        </div>
      </div>
      <div className="flex gap-4 mt-3 text-[10px] text-slate-400 justify-center">
        <span className="flex items-center gap-1">
          <span className="w-2.5 h-2.5 rounded-sm border border-amber-400/60 inline-block bg-amber-400/10" />
          Zone 14 (Golden Attacking Box)
        </span>
        <span className="flex items-center gap-1">
          <span className="w-2.5 h-2.5 rounded-sm border border-dashed border-sky-400/40 inline-block bg-sky-400/5" />
          Attacking Half-spaces
        </span>
      </div>
    </div>
  );
}
