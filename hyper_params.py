import subprocess, itertools, time

def main():
    start = time.time()

    lr = [ 0.0001 ]
    updates = [7500]
    adame = [0.00000001]
    batch = [16]
    replay = [10000]
    target = [50]
    
    hyperparams_set = list(itertools.product(*[lr, updates, adame, batch, replay, target]))
    print(hyperparams_set)
    for hp in hyperparams_set:
        #subprocess.call(['python','run.py', '-nogui' ,'-lr',str(hp[0]),'-updates',str(hp[1]),'-simlen','7199', '-netfp', 'single.net.xml', '-sumocfg', 'single.sumocfg',  '-lreps', str(hp[2]), '-batch', str(hp[3]), '-replay', str(hp[4]), '-target', str(hp[5]), '-actor', '1', '-learner', '1', '-vmin', str(hp[6]), '-load_weights', '-test' ])

        subprocess.call(['python','run.py' ,'-lr',str(hp[0]),'-updates',str(hp[1]),'-simlen','7200',  '-lre', str(hp[2]), '-batch', str(hp[3]), '-replay', str(hp[4]), '-target', str(hp[5]), '-actor', '7', '-learner', '1', '-nogui', '-eps', '0.05', '-sumocfg', 'double.sumocfg', '-netfp', 'double.net.xml' ])


    print('TOTAL HP SEARCH TIME')
    secs = time.time()-start
    print(str(int(secs/60.0))+' minutes ')

if __name__ == '__main__':
    main()
