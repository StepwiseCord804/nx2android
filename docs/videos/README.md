# Game videos

Short gameplay clips of ported games, one folder per game, for example `docs/videos/breath-of-the-wild/`.

GitHub refuses files over 100 MB and warns about files over 50 MB, and every video committed here stays in the repository's history for good, even after it is deleted. Phone screen recordings are much bigger than that (a 10-minute recording on the Galaxy S25 Ultra is about 1.2 GB), so:

- **Short clips (under about 50 MB):** shrink them first and commit them here. For example, 720p at 30 fps with [ffmpeg](https://ffmpeg.org/):

  ```
  ffmpeg -i Screen_Recording.mp4 -vf scale=-2:720 -r 30 -c:v libx264 -crf 28 -c:a aac -b:a 96k clip.mp4
  ```

  To cut out part of a recording first, add `-ss 00:01:30 -t 00:00:45` after `-i Screen_Recording.mp4` (start at 1:30, keep 45 seconds).

- **Long recordings:** attach them to a release instead (GitHub allows up to 2 GB a file there) or upload them to a video site, and link them from a `README.md` in the game's folder.

Videos only, no game files or keys.
