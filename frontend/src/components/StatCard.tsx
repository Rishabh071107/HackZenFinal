export default function StatCard({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="card !p-4">
      <div className="stat-value">{value}</div>
      <div className="stat-label mt-1">{label}</div>
    </div>
  );
}
