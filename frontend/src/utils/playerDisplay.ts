import type { Player } from "../api/client";

/** Squad jersey (1–11) when assigned; otherwise internal tracker id. */
export function playerSquadNumber(player: Pick<Player, "jersey_number" | "tracking_id">): number {
  return player.jersey_number ?? player.tracking_id;
}
