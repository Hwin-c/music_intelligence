const axios = require('axios');
require('dotenv').config(); // 환경 변수 불러오기

const API_BASE_URL = 'https://api.getsong.co'; // 변경된 도메인
const API_KEY = process.env.GETSONGBPM_API_KEY; // .env에서 API 키 읽기

/**
 * 특정 BPM의 곡을 검색
 * @param {number} bpm - 검색할 BPM
 * @param {number} limit - 결과 개수
 * @returns {Promise<Array>} - 곡 목록
 */
async function searchByBPM(bpm, limit = 3) {
  try {
    const response = await axios.get(`${API_BASE_URL}/tempo/`, {
      params: {
        api_key: API_KEY,
        bpm,
        limit,
      },
      headers: {
        'User-Agent': 'Mozilla/5.0' // 403 방지
      }
    });

    if (response.data && response.data.tempo) {
      return response.data.tempo;
    } else {
      throw new Error('No results found.');
    }
  } catch (error) {
    console.error('❌ 검색 실패:', error.response?.status || error.message);
    return [];
  }
}

module.exports = {
  searchByBPM,
};