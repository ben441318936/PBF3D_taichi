ffmpeg -framerate 60 -i ./render4/ply_render.mantra1.%4d.png -c:v libx264 -r 30 out4.mp4

ffmpeg -framerate 60 -i ./frames/image_%4d.png -c:v libx264 -r 30 out.mp4


ffmpeg -framerate 60 -i ./frames/%3d.png -c:v libx264 -r 30 out.mp4

ffmpeg -ss 00:00 -i out.mp4 -vf "fps=30,scale=320:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -loop 0 out.gif