function printSongs(songs) {
  if (!songs.length) {
    console.log('😥 추천 곡을 찾을 수 없습니다.');
    return;
  }

  console.log('🎵 추천 곡 목록:');
  songs.forEach((song, idx) => {
    console.log(`\n#${idx + 1}`);
    console.log(`제목: ${song.song_title}`);
    console.log(`아티스트: ${song.artist.name}`);
    console.log(`BPM: ${song.tempo}`);
    console.log(`앨범: ${song.album.title} (${song.album.year})`);
    console.log(`링크: ${song.song_uri}`);
  });
}

module.exports = {
  printSongs
};