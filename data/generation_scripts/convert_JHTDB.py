import numpy as np
import os
import scipy.ndimage

from pyJHTDB import libJHTDB
import pyJHTDB.dbinfo as dbinfo

outDir = "data/128_test/"
size = 128
seeds = range(561,1000)
#seeds = [1]

resFactors = [0.25, 0.5, 0.75, 1, 2, 3, 4]
iso = {"id":"isotropic1024coarse", "dt":180, "resWeight":[0.14,0.14,0.14, 0.16, 0.14,0.14,0.14], "randOff":0,  "info" : dbinfo.isotropic1024coarse}
cha = {"id":"channel",             "dt":37,  "resWeight":[0.14,0.14,0.14, 0.16, 0.14,0.14,0.14], "randOff":0,  "info" : dbinfo.channel}
mhd = {"id":"mhd1024",             "dt":95,  "resWeight":[0.14,0.14,0.14, 0.16, 0.14,0.14,0.14], "randOff":25, "info" : dbinfo.mhd1024}
tra = {"id":"transition_bl",       "dt":25,  "resWeight":[0.14,0.14,0.14, 0.30, 0.28,0,   0   ], "randOff":0,  "info" : dbinfo.transition_bl}
#rot = {"id":"rotstrat4096",        "dt":1,   "resWeight":[0.14,0.14,0.14, 0.16, 0.14,0.14,0.14], "randOff":125, "info" : dbinfo.rotstrat4096}
datasets = [iso, cha, mhd, tra]


lJHTDB = libJHTDB()
lJHTDB.initialize()
lJHTDB.add_token("ADD: personal JHTDB authorization token")

for seed in seeds:
    np.random.seed(seed)

    for d in datasets:
        nx = d["info"]["nx"]
        ny = d["info"]["ny"]
        nz = d["info"]["nz"]
        nt = np.array( d["info"]["time"]).shape[0]
        #print("Dataset %s: %dx%dx%d in %d frames" % (d["id"], nx,ny,nz,nt))

        if nt > 10*d["dt"]:
            time = np.random.randint(1, nt - (10*d["dt"]))
        elif d["id"] == "rotstrat4096":
            time = np.random.choice(np.array([1,2]))
        else:
            raise ValueError("%s -- Too few time frames!" % d["id"])

        resFac = np.random.choice(np.array(resFactors), p=d["resWeight"])
        if d["id"] == "transition_bl":
            start = np.array([
                np.random.randint(1,3000),
                1,
                np.random.randint(1, nz - resFac*size + 2),
            ])
        else:
            start = np.array([
                np.random.randint(1, nx - resFac*size + 2),
                np.random.randint(1, ny - resFac*size + 2),
                np.random.randint(1, nz - resFac*size + 2),
            ])
        end = np.random.permutation([resFac*size-1, resFac*size-1, resFac*size-1]) + start
        step = np.array([resFac, resFac, resFac])
        step = np.maximum(step, 1) #db can't be queried for resFac<1, instead get smaller cutout and interpolate

        print("jhtdb_%s/sim_%06d" % (d["id"], seed))
        outPath = os.path.join(outDir, "jhtdb_%s/sim_%06d" % (d["id"], seed))
        if not os.path.isdir(outPath):
            os.makedirs(outPath)
        f = open(os.path.join(outPath, "description.txt"), "w")

        results = []
        for s in range(11):
            if d["id"] == "rotstrat4096":
                timeAdj = time + int((s*d["dt"])/3)
            else:
                timeAdj = time + s*d["dt"]

            if d["randOff"] > 0:
                if d["id"] != "rotstrat4096" or (d["id"] == "rotstrat4096" and (s%3) != 0):
                    for i in [0,1,2]:
                        randOff = np.random.randint(0, d["randOff"])
                        if (start[i] - randOff) <= 0:
                            start[i] = start[i] + randOff
                            end[i] = end[i] + randOff
                        elif (end[i] + randOff) <= [nx,ny,nz][i]:
                            start[i] = start[i] - randOff
                            end[i] = end[i] - randOff
                        else:
                            if np.random.sample() > 0.5:
                                start[i] = start[i] + randOff
                                end[i] = end[i] + randOff
                            else:
                                start[i] = start[i] - randOff
                                end[i] = end[i] - randOff

            results += [ lJHTDB.getCutout( data_set=d["id"], field='u', time_step=timeAdj,
                        start=start.astype(np.int32), end=end.astype(np.int32), step=step.astype(np.int32), ).astype(np.float32) ]
            #print("Slice: %d, time: %d, start: (%d %d %d), end: (%d %d %d), step: (%d %d %d)" % (s, timeAdj, start[0], start[1], start[2], end[0], end[1], end[2], step[0], step[1], step[2]))
            f.write("Slice: %d, time: %d, start: (%d %d %d), end: (%d %d %d), step: (%d %d %d)\n" % (s, timeAdj, start[0], start[1], start[2], end[0], end[1], end[2], step[0], step[1], step[2]))

            dMax, dMin = ( np.max(results[-1]), np.min(results[-1]) )
            if dMax == dMin:
                print("Warning: slice of %s is empty!" % (outPath))
                #return

        f.close()
        combined = np.squeeze(np.stack(results, axis=0))
        if combined.shape[1] != size:
            #print("Rescaling %s" % outPath)
            zoom = [1, size / combined.shape[1], size / combined.shape[2], size / combined.shape[3], 1]
            combined = scipy.ndimage.zoom(combined, zoom, order=1)

        outFile = os.path.join(outPath, "velocity_000000.npz")
        np.savez_compressed(outFile, combined)

lJHTDB.finalize()
