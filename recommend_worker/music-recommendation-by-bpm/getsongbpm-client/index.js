require('dotenv').config(); // .env에서 API 키 불러오기
const { searchByBPM } = require('./api/getsongbpm'); // API 모듈
const { printSongs } = require('./utils/display');   // 출력 함수

const bpm = 120; // 찾고자 하는 BPM
const limit = 3; // 결과 개수

searchByBPM(bpm, limit)
  .then(printSongs)
  .catch((err) => console.error('❌ 에러 발생:', err));