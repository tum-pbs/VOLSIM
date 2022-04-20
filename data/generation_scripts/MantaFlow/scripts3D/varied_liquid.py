import sys, os, math, random, shutil, datetime

from manta import *
import numpy as np
sys.path.append(os.getcwd() + '/tensorflow/tools')
import paramhelpers as ph
import imageio
import random

withGUI = False
modes = ["noise",
         "dropPosX", "dropPosXNeg", "dropPosY", "dropPosYNeg", "dropPosZ", "dropPosZNeg",
         "dropRadius", "dropRadiusNeg",
         "gravityX", "gravityXNeg", "gravityY", "gravityYNeg", "gravityZ", "gravityZNeg"] # count: 15
res = 128
outputFolderTrain = "data/train_verbose"
outputFolderTest = "data/test_verbose"

# parse arguments
if len(sys.argv) == 5:
    mode = modes[int(sys.argv[1])]
    amount = int(sys.argv[2])
    seed = int(sys.argv[3])
    isTrain = sys.argv[4] != "Test"

    mantaMsg("--------------------------------------------")
    mantaMsg("| Mode: %s" % mode)
    mantaMsg("| Amount: %i" % amount)
    mantaMsg("| Seed: %i" % seed)
    mantaMsg("| Trainset: %s" % isTrain)
    mantaMsg("--------------------------------------------")
else:
    mode = modes[0]
    amount = 0
    seed = 0
    isTrain = True

    mantaMsg("Wrong parameters!")
    exit(1)


# solver params
gs = vec3(res,res,res)
s = Solver(name='main', gridSize = gs, dim=3)
s.timestep = 1.2

# prepare output folders, rendering and log
if isTrain:
    basepath = "%s/liquid_%s/sim_%06d" % (outputFolderTrain, mode, seed)
else:
    basepath = "%s/liquid_bgnoise_%s/sim_%06d" % (outputFolderTest, mode, seed)
if not os.path.exists(basepath + "/src"):
    os.makedirs(basepath + "/src")

log = {}
log["Timestamp"] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
log["Solver"] = {"Source File": os.path.basename(__file__),
                "Timestep": s.timestep, "Resolution": [gs.x, gs.y, gs.z], "Numpy Seed": seed}
log["Variation"] = {"Parameter" : mode, "Amount" : amount}
log["Stats"] = {"Levelset" : [], "Velocity" : [], "Flags" : []}

renderpath = basepath + "/render/"
if not os.path.exists(renderpath):
    os.makedirs(renderpath)
renderData  = {"phiMean" : [], "phiSlice" : [], "velMean" : [], "velSlice" : [], "flagsMean" : [], "flagsSlice" : [],}


# prepare grids
flags = s.create(FlagGrid)
flagsCopy = s.create(RealGrid)
vel = s.create(MACGrid)
velOld = s.create(MACGrid)
pressure = s.create(RealGrid)
tmpVec3  = s.create(VecGrid)
phi = s.create(LevelsetGrid)
phiCopy = s.create(LevelsetGrid)
noise = s.create(MACGrid)
mesh = s.create(Mesh)

# prepare particle system and acceleration structures
pp = s.create(BasicParticleSystem)
pVel = pp.create(PdataVec3)
pindex = s.create(ParticleIndexSystem) 
gpi    = s.create(IntGrid)

flags.initDomain(boundaryWidth=1)
phi.initFromFlags(flags)

if withGUI:
    gui = Gui()
    gui.show(True)
    #gui.pause()

# scene setup
center = vec3(int(gs.x*0.5), int(gs.y*0.1), int(gs.z*0.5))
size = vec3(int(res*0.5), int(gs.y*0.1), int(res*0.5))
pool = s.create(Box, center=center, size=size)
log["Liquid Box 1"] = {"Position" : [center.x, center.y, center.z], "Size" : [size.x, size.y, size.z]}

center = vec3(int(gs.x*(0.35+random.uniform(-0.05,0.05))), int(gs.y*0.4), int(gs.z*(0.35+random.uniform(-0.05,0.05))))
size = vec3(int(res*0.3), int(gs.y*0.2), int(res*0.3))
dam = s.create(Box, center=center, size=size)
log["Liquid Box 2"] = {"Position" : [center.x, center.y, center.z], "Size" : [size.x, size.y, size.z]}

center = vec3(int(gs.x*0.5), int(gs.y*0.75), int(gs.z*0.5))
radius = int(res*0.17)
if isTrain:
    if mode == "dropPosX":          center += vec3(2*amount,0,0)
    elif mode == "dropPosXNeg":     center -= vec3(2*amount,0,0)
    elif mode == "dropPosY":        center += vec3(1.3*amount,1.4*amount,0)
    elif mode == "dropPosYNeg":     center -= vec3(1.3*amount,1.5*amount,0)
    elif mode == "dropPosZ":        center += vec3(0,0,2*amount)
    elif mode == "dropPosZNeg":     center -= vec3(0,0,2*amount)
    elif mode == "dropRadius":      radius += 0.4*amount
    elif mode == "dropRadiusNeg":   radius -= 0.7*amount
else:
    if mode == "dropPosX":          center += vec3(3.5*amount,0,0)
    elif mode == "dropPosXNeg":     center -= vec3(3.5*amount,0,0)
    elif mode == "dropPosY":        center += vec3(2.7*amount,1.4*amount,0)
    elif mode == "dropPosYNeg":     center -= vec3(2.7*amount,1.5*amount,0)
    elif mode == "dropPosZ":        center += vec3(0,0,3.5*amount)
    elif mode == "dropPosZNeg":     center -= vec3(0,0,3.5*amount)
    elif mode == "dropRadius":      radius += 1.0*amount
    elif mode == "dropRadiusNeg":   radius -= 1.2*amount
drop = s.create(Sphere, center=center, radius=radius)
log["Liquid Drop"] = {"Position" : [center.x, center.y, center.z], "Radius" : radius} 

phi.join(pool.computeLevelset())
phi.join(dam.computeLevelset())
#phi.join(drop.computeLevelset())
flags.updateFromLevelset(phi)

#sampleFlagsWithParticles(flags=flags, parts=pp, discretization=2, randomness=0.2)
sampleLevelsetWithParticles(phi=phi, flags=flags, parts=pp, discretization=2, randomness=0.1)


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
for t in range(400):
    mantaMsg('\nFrame %i' % (s.frame))

    if t == 25:
        sampleLevelsetWithParticles(phi=drop.computeLevelset(), flags=flags, parts=pp, discretization=2, randomness=0.1)

    # FLIP 
    pp.advectInGrid(flags=flags, vel=vel, integrationMode=IntRK4, deleteInObstacle=False)
    mapPartsToMAC(vel=vel, flags=flags, velOld=velOld, parts=pp, partVel=pVel, weight=tmpVec3)
    extrapolateMACFromWeight(vel=vel, distance=2, weight=tmpVec3)
    markFluidCells(parts=pp, flags=flags)

    # resampling with surface level set
    gridParticleIndex(parts=pp, flags=flags, indexSys=pindex, index=gpi)
    improvedParticleLevelset(parts=pp, indexSys=pindex, flags=flags, index=gpi, phi=phi, radiusFactor=1.0)
    extrapolateLsSimple(phi=phi, distance=4, inside=True)
    #phi.createMesh(mesh)

    gravity = vec3(0,-1e-3,0)
    if isTrain:
        if mode == "gravityX":          gravity += vec3(0.000009*amount,0,0)
        elif mode == "gravityXNeg":     gravity -= vec3(0.000009*amount,0,0)
        elif mode == "gravityY":        gravity += vec3(0,0.000020*amount,0)
        elif mode == "gravityYNeg":     gravity -= vec3(0,0.000020*amount,0)
        elif mode == "gravityZ":        gravity += vec3(0,0,0.000011*amount)
        elif mode == "gravityZNeg":     gravity -= vec3(0,0,0.000011*amount)
    else:
        if mode == "gravityX":          gravity += vec3(0.000020*amount,0,0)
        elif mode == "gravityXNeg":     gravity -= vec3(0.000020*amount,0,0)
        elif mode == "gravityY":        gravity += vec3(0,0.000025*amount,0)
        elif mode == "gravityYNeg":     gravity -= vec3(0,0.000025*amount,0)
        elif mode == "gravityZ":        gravity += vec3(0,0,0.000020*amount)
        elif mode == "gravityZNeg":     gravity -= vec3(0,0,0.000020*amount)
    addGravity(flags=flags, vel=vel, gravity=0.5*gravity)
    log["Gravity"] = [gravity.x, gravity.y, gravity.z]

    if isTrain:
        if mode == "noise":
            noiseStrength = 0.08*amount
            noiseSeed = seed+1234
            noiseMode = "uniform"
        else:
            noiseStrength = 0.3
            #noiseStrength = 0.0
            noiseSeed = random.randint(0, 999999999)
            noiseMode = "normal"
    else:
        if mode == "noise":
            noiseStrength = 0.13*amount
            noiseSeed = seed+1234
            noiseMode = "uniform"
        else:
            #noiseStrength = 0.06
            noiseStrength = 0.0
            noiseSeed = random.randint(0, 999999999)
            noiseMode = "normal"
    createRandomField(noise=noise, strength=noiseStrength, bWidth=4, mode=noiseMode, seed=noiseSeed)
    vel.add(noise)
    log["Noise"] = {"Strength" : noiseStrength, "Seed" : noiseSeed, "Mode" : noiseMode}

    # pressure solve
    setWallBcs(flags=flags, vel=vel)
    solvePressure(flags=flags, vel=vel, pressure=pressure)
    setWallBcs(flags=flags, vel=vel)

    pVel.setSource( vel, isMAC=True )
    adjustNumber(parts=pp, vel=vel, flags=flags, minParticles=8, maxParticles=16, phi=phi, radiusFactor=1.0)


    mantaMsg("%s: %0.4f" % (mode, amount))

    # save data
    phiCopy.copyFrom(phi)
    extrapolateLsSimple(phi=phiCopy, distance=10, inside=True)
    extrapolateLsSimple(phi=phiCopy, distance=10, inside=False)

    # BACKGROUND NOISE
    if not isTrain:
        noise.clear()
        noiseStrength = 0.2
        noiseSeed = random.randint(0, 999999999)
        noiseMode = "normal"
        createRandomField(noise=noise, excludeGrid=flags, strength=noiseStrength, bWidth=4, mode=noiseMode, seed=noiseSeed)
        vel.add(noise)
        log["Background Noise"] = {"Strength" : noiseStrength, "Seed" : noiseSeed, "Mode" : noiseMode}

    velNP = np.zeros([res, res, res, 3])
    copyGridToArrayMAC( target=velNP, source=vel )
    renderData["velMean"].append( prepareRender(velNP, "mean") )
    renderData["velSlice"].append( prepareRender(velNP, "slice") )
    log["Stats"]["Velocity"].append( "Min:%0.8f Max:%0.8f Avg: %0.8f" % (np.min(velNP), np.max(velNP), np.mean(velNP)) )

    phiNP = np.zeros([res, res, res, 1])
    copyGridToArrayReal( target=phiNP, source=phiCopy )
    renderData["phiMean"].append( prepareRender(phiNP, "mean") )
    renderData["phiSlice"].append( prepareRender(phiNP, "slice") )
    log["Stats"]["Levelset"].append( "Min:%0.8f Max:%0.8f Avg: %0.8f" % (np.min(phiNP), np.max(phiNP), np.mean(phiNP)) )

    flagsNP = np.zeros([res, res, res, 1])
    copyGridToArrayInt( target=flagsNP, source=flags )
    renderData["flagsMean"].append( prepareRender(flagsNP, "mean") )
    renderData["flagsSlice"].append( prepareRender(flagsNP, "slice") )
    log["Stats"]["Flags"].append( "Min:%0.8f Max:%0.8f Avg: %0.8f" % (np.min(flagsNP), np.max(flagsNP), np.mean(flagsNP)) )

    #if t > 50:
    if t == 80:
        np.savez_compressed( "%s/velocity_%06d_part%02d.npz" % (basepath, t, amount), velNP.astype(np.float32) )
        if isTrain:
            np.savez_compressed( "%s/phi_%06d_part%02d.npz" % (basepath, t, amount), phiNP.astype(np.float32) )
            np.savez_compressed( "%s/flags_%06d_part%02d.npz" % (basepath, t, amount), flagsNP.astype(np.float32) )
        break
    #if t==110: break # used frame 80

    # FLIP velocity update
    extrapolateMACSimple(flags=flags, vel=vel)
    flipVelocityUpdate(vel=vel, velOld=velOld, flags=flags, parts=pp, partVel=pVel, flipRatio=0.97)

    s.step()


# save meta information
ph.writeParams(basepath + "/src/description%02d.json" % amount, log)
shutil.copy(os.path.abspath(__file__), basepath + "/src/%s" % os.path.basename(__file__))

for key in renderData.keys():
    outPath = "%s%s%02d.mp4" % (renderpath, key, amount)
    #imageio.mimwrite(outPath, renderData[key], quality=6, fps=10, output_params=['-codec:v', 'copy', outPath])
    imageio.mimwrite(outPath, renderData[key], quality=8, fps=10, ffmpeg_log_level="error")