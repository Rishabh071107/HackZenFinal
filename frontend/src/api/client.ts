import axios from "axios";

export const api = axios.create({ baseURL: "/api/v1" });

export interface Match {
  id: string;
  name: string;
  team_a_name: string;
  team_b_name: string;
  tournament?: string;
  status: string;
  duration_seconds?: number;
  total_frames?: number;
  avg_detection_confidence?: number;
  avg_tracking_accuracy?: number;
  confirmed_tracks?: number;
  id_switches?: number;
  recovered_tracks?: number;
  rejected_false_detections?: number;
  created_at: string;
}

export interface ProcessingJob {
  id: string;
  match_id: string;
  current_stage: string;
  progress_percent: number;
  error_message?: string;
}

export interface Team {
  id: string;
  match_id: string;
  label: string;
  display_name?: string;
  jersey_color_hex: string;
}

export interface Player {
  id: string;
  team_id: string;
  tracking_id: number;
  jersey_number?: number;
  estimated_role?: string;
  thumbnail_path?: string;
}

export interface AnalyticsPayload {
  id: string;
  scope: string;
  metrics: Record<string, any>;
  computed_at: string;
}

export interface HeatmapPayload {
  id: string;
  scope: string;
  image_path: string;
  grid_data_path?: string;
}

export interface TrajectoryPayload {
  id: string;
  player_id: string;
  points: [number, number, number][];
  start_point?: [number, number];
  end_point?: [number, number];
}

export interface ConsistencyLogEvent {
  frame_number: number;
  previous_track_id: number;
  matched_detection_index: number;
  iou_score: number;
  appearance_score: number;
  motion_score: number;
  geometry_score: number;
  final_cost: number;
  decision: string;
}

export interface ConsistencyLogPayload {
  match_id: string;
  confirmed_tracks: number;
  id_switches: number;
  recovered_tracks: number;
  rejected_false_detections: number;
  total_events: number;
  events: ConsistencyLogEvent[];
}

export interface PlayerFeaturesPayload {
  player_id: string;
  tracking_id: number;
  estimated_role?: string;
  team_label?: string;
  features: {
    color_features?: {
      lab_dominant_color?: number[];
      lab_mean?: number[];
      hsv_hue_bincount?: number[];
      hsv_histogram?: number[];
    };
    shape_features?: {
      aspect_ratio?: number;
      area_ratio?: number;
      hu_moments?: number[];
    };
    texture_features?: {
      lbp_histogram_10bin?: number[];
      lbp_histogram?: number[];
    };
    edge_features?: {
      canny_edge_density?: number;
      quadrants_distribution?: number[];
      edge_quadrants?: number[];
    };
    geometric_features?: {
      pitch_x_m?: number;
      pitch_y_m?: number;
    };
    landmark_features?: {
      body_keypoints_5pt?: { name: string; norm_x: number; norm_y: number }[];
      keypoints?: number[][];
    };
  };
}

export const MatchesAPI = {
  list: () => api.get<Match[]>("/matches").then((r) => r.data),
  get: (id: string) => api.get<Match>(`/matches/${id}`).then((r) => r.data),
  upload: (form: FormData) =>
    api.post<Match>("/matches/upload", form, {
      headers: { "Content-Type": "multipart/form-data" },
    }).then((r) => r.data),
  delete: (id: string) => api.delete<{ status: string; message: string }>(`/matches/${id}`).then((r) => r.data),
  analyze: (id: string) => api.post<ProcessingJob>(`/matches/${id}/analyze`).then((r) => r.data),
  status: (id: string) => api.get<ProcessingJob>(`/matches/${id}/status`).then((r) => r.data),
  teams: (id: string) => api.get<Team[]>(`/matches/${id}/teams`).then((r) => r.data),
  analytics: (id: string) => api.get<AnalyticsPayload>(`/matches/${id}/analytics`).then((r) => r.data),
  heatmap: (id: string) => api.get<HeatmapPayload>(`/matches/${id}/heatmap`).then((r) => r.data),
  consistencyLog: (id: string) => api.get<ConsistencyLogPayload>(`/matches/${id}/consistency-log`).then((r) => r.data),
};

export const TeamsAPI = {
  get: (teamId: string) => api.get<Team>(`/teams/${teamId}`).then((r) => r.data),
  players: (teamId: string) => api.get<Player[]>(`/teams/${teamId}/players`).then((r) => r.data),
  analytics: (teamId: string) => api.get<AnalyticsPayload>(`/teams/${teamId}/analytics`).then((r) => r.data),
  heatmap: (teamId: string) => api.get<HeatmapPayload>(`/teams/${teamId}/heatmap`).then((r) => r.data),
};

export const PlayersAPI = {
  get: (playerId: string) => api.get<Player>(`/players/${playerId}`).then((r) => r.data),
  analytics: (playerId: string) => api.get<AnalyticsPayload>(`/players/${playerId}/analytics`).then((r) => r.data),
  heatmap: (playerId: string) => api.get<HeatmapPayload>(`/players/${playerId}/heatmap`).then((r) => r.data),
  trajectory: (playerId: string) => api.get<TrajectoryPayload>(`/players/${playerId}/trajectory`).then((r) => r.data),
  features: (playerId: string) => api.get<PlayerFeaturesPayload>(`/players/${playerId}/features`).then((r) => r.data),
};
