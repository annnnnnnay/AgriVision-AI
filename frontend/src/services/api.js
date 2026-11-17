import axios from "axios";
import { supabase } from "./supabase";

const API_BASE_URL = import.meta.env.VITE_API_URL || "";
const PREDICTION_TIMEOUT_MS = 110000;

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    "Content-Type": "application/json",
  },
});

// Request interceptor — add auth token
api.interceptors.request.use(
  async (config) => {
    let token = localStorage.getItem("token");
    if (supabase) {
      const { data, error } = await supabase.auth.getSession();
      if (error) throw error;
      token = data.session?.access_token || null;
    }

    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    } else {
      delete config.headers.Authorization;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Response interceptor — handle errors
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.code === "ECONNABORTED" || error.code === "ETIMEDOUT") {
      return Promise.reject(
        new Error(
          "Image analysis timed out. Please retry with a smaller image or check the vision provider configuration."
        )
      );
    }

    const message =
      error.response?.data?.error?.message ||
      error.response?.data?.message ||
      error.message ||
      "An unexpected error occurred";
    return Promise.reject(new Error(message));
  }
);

/**
 * Upload a plant image for disease prediction.
 */
export async function predictDisease(
  imageFile,
  language = "en",
  imageType = "unknown",
  cropHint = ""
) {
  const formData = new FormData();
  formData.append("file", imageFile);
  formData.append("language", language);
  formData.append("image_type", imageType);
  if (cropHint.trim()) formData.append("crop_hint", cropHint.trim());

  const response = await api.post("/api/v1/predict", formData, {
    headers: { "Content-Type": "multipart/form-data" },
    timeout: PREDICTION_TIMEOUT_MS,
  });

  return response.data;
}

/**
 * Get list of supported crops.
 */
export async function getCrops() {
  const response = await api.get("/api/v1/crops");
  return response.data;
}

/**
 * Get disease info by ID.
 */
export async function getDiseaseInfo(diseaseId, language = "en") {
  const response = await api.get(`/api/v1/diseases/${diseaseId}`, {
    params: { language },
  });
  return response.data;
}

/**
 * Health check.
 */
export async function healthCheck() {
  const response = await api.get("/api/health");
  return response.data;
}

export default api;
