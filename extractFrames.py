import ffmpeg

YOUR_FILE = 'chess1.MOV'
probe = ffmpeg.probe(YOUR_FILE)
time = float(probe['streams'][0]['duration']) // 2
width = probe['streams'][0]['width']

parts = int(time)
intervals = time // parts
intervals = int(intervals)
interval_list = [(i * intervals, (i + 1) * intervals) for i in range(parts)]
i = 0

for item in interval_list:
    (
        ffmpeg
        .input(YOUR_FILE, ss=item[1])
        .filter('scale', width, -1)
        .output('Image' + str(i) + '.jpg', vframes=1)
        .run()
    )
    i += 1
    



