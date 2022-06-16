import gdown

datasets = {
    'data_div2k.npz':'115l577qCHEbGN-GzE0MpnyHFoGJDy5Xw',
    'video_16_128.npz':'1-5TOdbH4j6Fr4TV1bqppo9H5JgAhvJUs',
}

for name in datasets:
    gdown.download(id=datasets[name], output=name, quiet=False)