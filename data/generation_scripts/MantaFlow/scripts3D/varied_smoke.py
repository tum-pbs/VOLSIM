import sys, os, math, random, shutil, datetime

from manta import *
import numpy as np
sys.path.append(os.getcwd() + '/tensorflow/tools')
import paramhelpers as ph
import imageio

withGUI = False
modes = ["noise", "sourcePosX", "sourcePosY",
         "buoyancyX", "buoyancyY", "buoyancyYNeg",
         "obsPosX", "obsPosY", "obsPosYNeg",
         "obsRadius", "obsRadiusNeg",
         "obsForceX", "obsForceY", "obsForceYNeg",
         "obsForceRotX", "obsForceRotZ"] # count: 16
res = 128
outputFolder = "data/train_verbose"

# parse arguments
if len(sys.argv) == 4:
    mode = modes[int(sys.argv[1])]
    amount = int(sys.argv[2])
    seed = int(sys.argv[3])

    mantaMsg("--------------------------------------------")
    mantaMsg("| Mode: %s" % mode)
    mantaMsg("| Amount: %i" % amount)
    mantaMsg("| Seed: %i" % seed)
    mantaMsg("--------------------------------------------")
else:
    mode = modes[0]
    amount = 0
    seed = 0

    mantaMsg("Wrong parameters!")
    exit(1)


# solver params
gs = vec3(res,res,res)
s = Solver(name='main', gridSize = gs, dim=3)
s.timestep = 1.2

# prepare output folders, rendering and log
basepath = "%s/smoke_%s/sim_%06d" % (outputFolder, mode, seed)
if not os.path.exists(basepath + "/src"):
    os.makedirs(basepath + "/src")

log = {}
log["Timestamp"] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
log["Solver"] = {"Source File": os.path.basename(__file__),
                "Timestep": s.timestep, "Resolution": [gs.x, gs.y, gs.z], "Numpy Seed": seed}
log["Variation"] = {"Parameter" : mode, "Amount" : amount}
log["Stats"] = {"Velocity" : [], "Density" : [], "Pressure" : []}

renderpath = basepath + "/render/"
if not os.path.exists(renderpath):
    os.makedirs(renderpath)
renderData  = {"densMean" : [], "densSlice" : [], "velMean" : [], "velSlice" : [], "presMean" : [], "presSlice" : [],}


# prepare grids
flags = s.create(FlagGrid)
vel = s.create(MACGrid)
noise = s.create(MACGrid)
density = s.create(RealGrid)
pressure = s.create(RealGrid)
obsForce = s.create(VecGrid)

bWidth=1
flags.initDomain(boundaryWidth=bWidth) 
flags.fillGrid()

setOpenBound(flags, bWidth,'yY',FlagOutflow|FlagEmpty) 

if withGUI:
    gui = Gui()
    gui.show( True )
    #gui.pause()

# scene setup
center = vec3(int(gs.x*0.5), int(gs.y*0.1), int(gs.z*0.5))
radius = int(res*0.14)
height = vec3(0, int(gs.y*0.02), 0)
if mode == "sourcePosX":        center += vec3(1*amount,0,0)
elif mode == "sourcePosY":      center += vec3(0,1.5*amount,0)
source = s.create(Cylinder, center=center, radius=radius, z=height)
log["Smoke Source"] = {"Position" : [center.x, center.y, center.z], "Radius" : radius, "Height" : height.y} 

center = vec3(int(gs.x*0.5), int(gs.y*0.5), int(gs.z*0.5))
radius = int(res*0.1)
if mode == "obsPosX":           center += vec3(1*amount,0,0)
elif mode == "obsPosY":         center += vec3(0,2*amount,0)
elif mode == "obsPosYNeg":      center -= vec3(0,2*amount,0)
elif mode == "obsRadius":       radius += 0.7*amount
elif mode == "obsRadiusNeg":    radius -= 0.7*amount
obs = s.create(Sphere, center=center, radius=radius)
obsPhi = obs.computeLevelset()

force = vec3(0, -1, 0)
if mode == "obsForceX":         force += vec3(0.35*amount,0,0)
elif mode == "obsForceY":       force += vec3(0,0.08*amount,0)
elif mode == "obsForceYNeg":    force -= vec3(0,0.18*amount,0)
elif mode == "obsForceRotX":    force = vec3(0, -math.cos(math.radians(9*amount)), -math.sin(math.radians(9*amount)))
elif mode == "obsForceRotZ":    force = vec3(math.sin(math.radians(9*amount)), -math.cos(math.radians(9*amount)), 0)
obsForce.setConst(0.025 * force)
log["Obstacle Force Field"] = {"Position" : [center.x, center.y, center.z], "Radius" : radius, "Force" : [force.x, force.y, force.z]}


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

    #source.applyToGrid(grid=density, value=1)
    applyShapeRandomized(grid=density, value=1, shape=source, strength=-0.15, seed=seed)

    advectSemiLagrange(flags=flags, vel=vel, grid=density, order=2, clampMode=2) 
    advectSemiLagrange(flags=flags, vel=vel, grid=vel,     order=2, clampMode=2)
    resetOutflow(flags=flags,real=density) 

    setWallBcs(flags=flags, vel=vel)
    gravity = vec3(0,-1e-3,0)
    if mode == "buoyancyX":     gravity += vec3(0.000022*amount,0,0)
    elif mode == "buoyancyY":   gravity += vec3(0,0.000017*amount,0)
    elif mode == "buoyancyYNeg":gravity -= vec3(0,-0.000022*amount,0)
    addBuoyancy(density=density, vel=vel, gravity=0.5*gravity, flags=flags)
    log["Buoyancy"] = [gravity.x, gravity.y, gravity.z]

    if mode == "noise":
        noiseStrength = 0.017*amount
        noiseSeed = seed+1234
        noiseMode = "uniform"
    else:
        noiseStrength = 0.125
        noiseSeed = random.randint(0, 999999999)
        noiseMode = "normal"
    createRandomField(noise=noise, strength=noiseStrength, excludeShape=obs, bWidth=2, mode=noiseMode, seed=noiseSeed)
    vel.add(noise)
    log["Noise"] = {"Strength" : noiseStrength, "Seed" : noiseSeed, "Mode" : noiseMode}

    addForceField(flags=flags, vel=vel, force=obsForce, region=obsPhi) 
    solvePressure(flags=flags, vel=vel, pressure=pressure, cgMaxIterFac=99, cgAccuracy=1e-04, zeroPressureFixing=True, preconditioner = PcMGStatic)


    mantaMsg("%s: %0.4f" % (mode, amount))

    # save data
    velNP = np.zeros([res, res, res, 3])
    copyGridToArrayMAC( target=velNP, source=vel )
    renderData["velMean"].append( prepareRender(velNP, "mean") )
    renderData["velSlice"].append( prepareRender(velNP, "slice") )
    log["Stats"]["Velocity"].append( "Min:%0.8f Max:%0.8f Avg: %0.8f" % (np.min(velNP), np.max(velNP), np.mean(velNP)) )

    densNP = np.zeros([res, res, res, 1])
    copyGridToArrayReal( target=densNP, source=density )
    renderData["densMean"].append( prepareRender(densNP, "mean") )
    renderData["densSlice"].append( prepareRender(densNP, "slice") )
    log["Stats"]["Density"].append( "Min:%0.8f Max:%0.8f Avg: %0.8f" % (np.min(densNP), np.max(densNP), np.mean(densNP)) )

    presNP = np.zeros([res, res, res, 1])
    copyGridToArrayReal( target=presNP, source=pressure )
    renderData["presMean"].append( prepareRender(presNP, "mean") )
    renderData["presSlice"].append( prepareRender(presNP, "slice") )
    log["Stats"]["Pressure"].append( "Min:%0.8f Max:%0.8f Avg: %0.8f" % (np.min(presNP), np.max(presNP), np.mean(presNP)) )

    #if t > 50:
    if t == 120:
        np.savez_compressed( "%s/velocity_%06d_part%02d.npz" % (basepath, t, amount), velNP.astype(np.float32) )
        np.savez_compressed( "%s/density_%06d_part%02d.npz" % (basepath, t, amount), densNP.astype(np.float32) )
        np.savez_compressed( "%s/pressure_%06d_part%02d.npz" % (basepath, t, amount), presNP.astype(np.float32) )
        break
    #if t==130:  break # used frame 120

    s.step()


# save meta information
ph.writeParams(basepath + "/src/description%02d.json" % amount, log)
shutil.copy(os.path.abspath(__file__), basepath + "/src/%s" % os.path.basename(__file__))

for key in renderData.keys():
    outPath = "%s%s%02d.mp4" % (renderpath, key, amount)
    #imageio.mimwrite(outPath, renderData[key], quality=6, fps=10, output_params=['-codec:v', 'copy', outPath])
    imageio.mimwrite(outPath, renderData[key], quality=8, fps=10, ffmpeg_log_level="error")
