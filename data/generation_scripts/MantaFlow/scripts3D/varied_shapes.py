import sys, os, random, shutil, datetime

from manta import *
import numpy as np
sys.path.append(os.getcwd() + '/tensorflow/tools')
import paramhelpers as ph
import imageio

withGUI = False
modes = ["pos", "pos_s", "pos_n", "pos_sn"] # count: 4
modesWave = ["pos", "pos_n"] # count: 2
res = 128
outputFolder = "data/test_verbose"

# parse arguments
if len(sys.argv) == 4:
    seed = int(sys.argv[2])
    useWaves = sys.argv[3] == "Waves"
    mode = modes[int(sys.argv[1])] if not useWaves else modesWave[int(sys.argv[1])]

    mantaMsg("--------------------------------------------")
    mantaMsg("| Mode: %s" % mode)
    mantaMsg("| Seed: %i" % seed)
    mantaMsg("| Use Waves: %s" % useWaves)
    mantaMsg("--------------------------------------------")
else:
    mode = modesWave[0]
    seed = 0
    useWaves = True

    mantaMsg("Wrong parameters!")
    exit(1)


# solver params
gs = vec3(res,res,res)
s = Solver(name='main', gridSize = gs, dim=3)
s.timestep = 1.0

# prepare output folders, rendering and log
if not useWaves:
    basepath = "%s/shapes_%s/sim_%06d" % (outputFolder, mode, seed)
else:
    basepath = "%s/waves_%s/sim_%06d" % (outputFolder, mode, seed)
if not os.path.exists(basepath + "/src"):
    os.makedirs(basepath + "/src")

log = {}
log["Timestamp"] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
log["Solver"] = {"Source File": os.path.basename(__file__),
                "Timestep": s.timestep, "Resolution": [gs.x, gs.y, gs.z], "Numpy Seed": seed}
if   mode == "pos": var = "position"
elif mode == "pos_s": var = "position (smoothed)"
elif mode == "pos_n": var = "position (with noise)"
elif mode == "pos_sn": var = "position (smoothed, with noise)"
log["Variation"] = {"Parameter" : var}

renderpath = basepath + "/render/"
if not os.path.exists(renderpath):
    os.makedirs(renderpath)
renderData  = {"flagsMean" : []}


# prepare grids
shapeGrid = s.create(RealGrid)
noise     = s.create(RealGrid)
flags     = s.create(FlagGrid) # unused! dummy for gui
vel       = s.create(VecGrid)  # unused! dummy for gui

flags.fillGrid()

# setup shapes
numShapes = np.random.randint(1,6) if not useWaves else np.random.randint(1,4)
shapeTypes = np.random.rand(numShapes)

starts = np.zeros((numShapes, 3))
ends = np.zeros((numShapes, 3))
for i in range(numShapes):
    while True:
        starts[i] = res * np.random.rand(3)
        ends[i] = res * np.random.rand(3)

        # clamp away from boundary by maximum possible radius
        dist = (0.25)*np.linalg.norm(ends[i]-starts[i])
        starts[i] = np.maximum(starts[i], dist)
        starts[i] = np.minimum(starts[i], res-dist)
        ends[i] = np.maximum(ends[i], dist)
        ends[i] = np.minimum(ends[i], res-dist)

        if np.linalg.norm(ends[i]-starts[i]) > 0.3*res:
            break


moveDirs = ends - starts
distances = np.linalg.norm(moveDirs, axis=1)
mantaMsg(str(numShapes))
mantaMsg(str(distances))
radii = np.zeros(numShapes)
for i in range(numShapes):
    if not useWaves:
        radii[i] = np.random.randint(0.1*distances[i], 0.2*distances[i])
        shapeType = "Sphere" if shapeTypes[i] > 0.5 else "Box"
    else:
        radii[i] = np.random.randint(0.4*distances[i], 0.8*distances[i])
        shapeType = "Wave"
    log["Shape " + str(i)] = {"Type": shapeType, "Start": list(starts[i]), "End": list(ends[i]), "Radius": radii[i]}

if withGUI:
    gui = Gui()
    gui.show( True )


# helper function for mp4 export
def prepareRender(data, mode):
    assert mode in ["mean", "slice"]
    if mode == "mean":
        data = np.mean(data, axis=0)
    else:
        data = data[int(res/2)]
    data = data - np.min(data)
    data = data / np.max(data)
    data = 255*data
    data = np.flip(data, axis=0)
    return data.astype(np.uint8)


#main loop
amounts = range(0,11)
for amount in amounts:
    mantaMsg('%s: %0.4f' % (mode, amount))

    shapeGrid.setConst(0.)
    for i in range(numShapes):
        frac = float(amount) / float(max(amounts))
        pos = vec3(starts[i,0], starts[i,1], starts[i,2]) + vec3(frac*moveDirs[i,0], frac*moveDirs[i,1], frac*moveDirs[i,2])

        if not useWaves:
            if shapeTypes[i] > 0.5:
                shapeObj = s.create(Sphere, center=pos, radius=radii[i])
            else:
                shapeObj = s.create(Box, center=pos, size=vec3(0.85*radii[i], 0.85*radii[i], 0.85*radii[i]))

            if mode == "pos_s" or mode == "pos_sn":
                shapeObj.applyToGridSmooth(shapeGrid, value=1)
            else:
                shapeObj.applyToGrid(shapeGrid, value=1)
        else:
            applyWaveToGrid(shapeGrid, center=pos, radius=radii[i], waviness=0.2*shapeTypes[i])

    if mode == "pos_n" or mode == "pos_sn":
        if not useWaves:
            noiseStrength = 0.25
        else:
            noiseStrength = 0.10
        noiseSeed = random.randint(0, 999999999)
        noiseMode = "normal"
        createRandomField(noise=noise, strength=noiseStrength, bWidth=0, mode=noiseMode, seed=noiseSeed)
        shapeGrid.add(noise)
        log["Noise"] = {"Strength" : noiseStrength, "Seed" : noiseSeed, "Mode" : noiseMode}

    # save data
    flagsNP = np.zeros([res, res, res, 1])
    copyGridToArrayReal( target=flagsNP, source=shapeGrid )
    np.savez_compressed( "%s/flags_%06d_part%02d.npz" % (basepath, 0, amount), flagsNP.astype(np.float32) )
    renderData["flagsMean"].append( prepareRender(flagsNP, "mean") )

    s.step()


# save meta information
ph.writeParams(basepath + "/src/description.json", log)
shutil.copy(os.path.abspath(__file__), basepath + "/src/%s" % os.path.basename(__file__))

for key in renderData.keys():
    if not useWaves:
        outPath = "%sshapes_%s_sim_%06d.mp4" % (renderpath, mode, seed)
    else:
        outPath = "%swaves_%s_sim_%06d.mp4" % (renderpath, mode, seed)
    #imageio.mimwrite(outPath, renderData[key], quality=6, fps=10, output_params=['-codec:v', 'copy', outPath])
    imageio.mimwrite(outPath, renderData[key], quality=8, fps=2, ffmpeg_log_level="error")