import numpy as np
import os, shutil
import imageio

baseDir = "data/train_verbose"
outDir = "data/train"
#baseDir = "data/test_verbose"
#outDir = "data/test"
outDirVidCopy = "data/videos"
combineVidsAll = {"smoke" : ["densMean", "densSlice", "velMean", "velSlice", "presMean", "presSlice"],
                "liquid": ["flagsMean", "flagsSlice", "velMean", "velSlice", "phiMean", "phiSlice"] }
convertData = True
processVid = True
copyVidOnly = False
ignoreTop = ["shapes", "waves"]
ignoreSim = []
ignoreFrameDict = {}
excludeIgnoreFrame = False

topDirs = os.listdir(baseDir)
topDirs.sort()
#shutil.rmtree(outDir)
#os.makedirs(outDir)

# top level folders
for topDir in topDirs:
    mantaMsg("\n" + topDir)
    if ignoreTop and any( item in topDir for item in ignoreTop ) :
        mantaMsg("Ignored")
        continue

    simDir = os.path.join(baseDir, topDir)
    sims = os.listdir(simDir)
    sims.sort()
    # sim_000000 folders
    for sim in sims:
        if ignoreSim and any( item in sim for item in ignoreSim ) :
            mantaMsg(sim + " - Ignored")
            continue

        currentDir = os.path.join(simDir, sim)
        files = os.listdir(currentDir)
        files.sort()

        destDir = os.path.join(outDir, topDir, sim)
        #if os.path.isdir(destDir):
        #    shutil.rmtree(destDir)
        if not os.path.isdir(destDir):
            os.makedirs(destDir)
        
        # single files
        for file in files:
            filePath = os.path.join(currentDir, file)

            # copy src folder to destination
            if os.path.isdir(filePath) and file == "src":
                dest = os.path.join(destDir, "src")
                if not os.path.isdir(dest):
                    shutil.copytree(filePath, dest, symlinks=False)


            # combine video files
            elif os.path.isdir(filePath) and file == "render":
                if not processVid:
                    continue

                dest = os.path.join(destDir, "render")
                if copyVidOnly:
                    shutil.copytree(filePath, dest, symlinks=False)
                    continue

                if not os.path.isdir(dest):
                    os.makedirs(dest)

                #mantaMsg(file)
                renderDir = os.path.join(currentDir, "render")
                vidFiles = os.listdir(renderDir)
                
                if "smoke" in topDir:       combineVids = combineVidsAll["smoke"]
                elif "liquid" in topDir:    combineVids = combineVidsAll["liquid"]
                else: combineVids = [""]

                for vidFile in vidFiles:
                    if combineVids[0] + "00.mp4" not in vidFile:
                        continue
                    
                    vidLine = []
                    for combineVid in combineVids:
                        # find all video part files corresponding to current one
                        vidParts = []
                        i = 0
                        while os.path.exists(os.path.join(renderDir, vidFile.replace(combineVids[0]+"00.mp4", combineVid+"%02d.mp4" % i))):
                            vidParts.append(vidFile.replace(combineVids[0]+"00.mp4", combineVid+"%02d.mp4" % i))
                            i += 1
                        assert len(vidParts) == 11
                        # combine each video part file
                        loadedVids = []
                        for part in vidParts:
                            currentFile = os.path.join(renderDir, part)
                            loaded = imageio.mimread(currentFile)
                            #mantaMsg(len(loaded))
                            #mantaMsg(loaded[0].shape)
                            loadedVids.append(loaded)
                        #temp1 = np.concatenate(loadedVids[0:4], axis=2)
                        #temp2 = np.concatenate(loadedVids[4:8], axis=2)
                        #temp3 = np.concatenate(loadedVids[8:11]+[np.zeros_like(loadedVids[0])], axis=2)
                        #vidLine.append(np.concatenate([temp1, temp2, temp3], axis=1))
                        vidLine.append(np.concatenate(loadedVids, axis=2))
                    combined = np.concatenate(vidLine, axis=1)

                    # save combined file
                    if combineVids[0] == "": newName = os.path.join(dest, "%s_%s_%s.mp4" % (topDir, sim, vidFile.replace("00.mp4", ".mp4")))
                    else:                    newName = os.path.join(dest, "%s_%s.mp4" % (topDir, sim))
                    imageio.mimwrite(newName, combined, quality=6, fps=11, ffmpeg_log_level="error")
                    # save copy
                    if combineVids[0] == "": newNameCopy = os.path.join(outDirVidCopy, "%s_%s_%s.mp4" % (topDir, sim, vidFile.replace("00.mp4", ".mp4")))
                    else:                    newNameCopy = os.path.join(outDirVidCopy, "%s_%s.mp4" % (topDir, sim))
                    imageio.mimwrite(newNameCopy, combined, quality=6, fps=11, ffmpeg_log_level="error")


            # copy description files to destination
            elif os.path.splitext(filePath)[1] == ".json" or os.path.splitext(filePath)[1] == ".py" or os.path.splitext(filePath)[1] == ".log":
                shutil.copy(filePath, destDir)


            # ignore other dirs and non .npz files
            elif os.path.isdir(filePath) or os.path.splitext(filePath)[1] != ".npz" or "part00" not in file:
                continue

            # combine part files
            else:
                if not convertData:
                    continue

                if ignoreFrameDict:
                    filterFrames = []
                    for key, value in ignoreFrameDict.items():
                        if key in topDir:
                            filterFrames = value
                            break
                    assert (filterFrames != []), "Keys in filterFrameDict don't match dataDir structure!"
                    
                    # continue for frames when excluding or including according to filter
                    if excludeIgnoreFrame == any( item in file for item in filterFrames ):
                        continue

                # find all part files corresponding to current one
                parts = [file]
                i = 1
                while os.path.exists(os.path.join(currentDir, file.replace("part00", "part%02d" % i))):
                    parts.append(file.replace("part00", "part%02d" % i))
                    i += 1
                assert len(parts) == 11
                # combine each part file
                domain = np.load(os.path.join(currentDir, parts[0]))['arr_0']
                res = domain.shape[0]
                combined = np.zeros([len(parts), res, res, res, domain.shape[3]])
                for f in range(len(parts)):
                    currentFile = os.path.join(currentDir, parts[f])
                    loaded = np.load(currentFile)['arr_0']
                    combined[f] = loaded
                
                # save combined file
                newName = file.replace("_part00", "")
                np.savez_compressed( os.path.join(destDir, newName), combined )
                loaded = np.load( os.path.join(destDir, newName) )['arr_0']
                mantaMsg(os.path.join(sim, newName) + "\t" + str(loaded.shape))
