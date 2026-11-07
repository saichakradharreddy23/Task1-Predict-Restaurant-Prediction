/**
 * RESTAURANT INSIGHT – Production API Configuration
 * Centralized endpoint mapping connecting Vercel frontend with Render backend.
 */
const API_BASE_URL = "https://task1-predict-restaurant-prediction.onrender.com";

/**
 * Returns absolute backend URL for any endpoint path
 * @param {string} endpoint - API route (e.g. '/api/health')
 * @returns {string} Fully qualified URL
 */
function getApiUrl(endpoint) {
  const path = endpoint.startsWith('/') ? endpoint : `/${endpoint}`;
  return `${API_BASE_URL}${path}`;
}
