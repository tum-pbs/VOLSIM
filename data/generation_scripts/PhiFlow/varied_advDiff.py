import os, sys, time, json, datetime
import numpy as np
import imageio

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from phi.tf.flow import *
from varied_sim_utils import *


withGUI = False
modes = ["f1", "f1neg", "f2", "f2neg", "f3", "f3neg", "f4", "f4neg", "f5", "f5neg", "f7", "f7neg",
        "o1", "o1neg", "o2", "o2neg", "od", "odneg", "noise",] # count: 19

# solver params
res = 128
batch = 11
dim = 3
destTrain = "data/train/"
destTest = "data/test/"
destVidCopy = "data/videos/"


@struct.definition()
class AdvDiff(DomainState):

    def __init__(self, domain, density=0, noiseScalar=0, velocity=0, force=0, noise=0, diffusivity=0,
                 tags=('advDiff', 'velocityfield'), name='advdiff', **kwargs):
        DomainState.__init__(self, **struct.kwargs(locals()))

    @struct.variable(default=0.0, dependencies=DomainState.domain)
    def density(self, density):
        return self.centered_grid('density', density)

    @struct.variable(default=0.0, dependencies=DomainState.domain)
    def noiseScalar(self, noiseScalar):
        return self.centered_grid('noiseScalar', noiseScalar)

    @struct.variable(default=0.0, dependencies=DomainState.domain)
    def velocity(self, velocity):
        return self.centered_grid('velocity', velocity, self.rank)

    @struct.variable(default=0.0, dependencies=DomainState.domain)
    def force(self, force):
        return self.centered_grid('force', force, self.rank)

    @struct.variable(default=0.0, dependencies=DomainState.domain)
    def noise(self, noise):
        return self.centered_grid('noise', noise, self.rank)

    @struct.constant(default=0.0)
    def diffusivity(self, diffusivity):
        return diffusivity


class AdvDiffPhysics(Physics):

    def __init__(self):
        Physics.__init__(self, [StateDependency('effects', 'velocity_effect', blocking=True)])

    def step(self, state, dt=1.0, effects=()):
        nu = state.diffusivity

        vel = state.velocity
        dens = state.density
        force = state.force
        noise = state.noise
        noiseScalar = state.noiseScalar

        dens = advect.semi_lagrangian(dens, vel, dt)
        dens = diffuse(dens, dt * nu, substeps=1)

        vel = vel + force
        vel = vel + noise
        dens = dens + noiseScalar

        for effect in effects:
            vel = effect_applied(effect, vel, dt)

        return state.copied_with(density=dens, velocity=vel, age=state.age + 1)


class AdvDiffSim(App):

    def __init__(self, mode, seed, isTrain, basename, basepath):
        App.__init__(self, basename, base_dir=destTrain if isTrain else destTest)
        self.mode = mode
        self.seed = seed
        self.isTrain = isTrain
        self.basename = basename
        self.basepath = basepath
        self.renderData = []
        self.log = {}

        state = AdvDiff( Domain((res, res, res), boundaries=PERIODIC), batch_size=batch,
                        velocity=Noise(channels=dim)*0, force=Noise(channels=dim)*0, noise=Noise(channels=dim)*0 )
        self.advDiff = world.add(state, physics=AdvDiffPhysics())
        self.setupSim()

        self.add_field('Density', lambda: self.advDiff.density)
        self.add_field('Velocity', lambda: self.advDiff.velocity)
        self.add_field('Force', lambda: self.advDiff.force)
        self.add_field('Noise', lambda: self.advDiff.noise)

    def action_reset(self):
        self.setupSim()
        self.steps = 0
        self.advDiff.age = 0

    def setupSim(self):
        #np.random.seed(self.seed)
        if self.isTrain:
            self.params = generateParams(self.mode, batch, dim, 0.025,
                                        0.021, 0.010, 0.020, 0.035, 0.040, 0.300,
                                        0.25, 0.15, 0.10, 0.005)
        else:
            self.params = generateParams(self.mode, batch, dim, 0.8,
                                        0.020, 0.012, 0.017, 0.028, 0.050, 0.500,
                                        0.12, 0.20, 0.15, 1.0)

        self.params["noise"] = self.params["noise"][:,None,None,None,None].repeat(res,1).repeat(res,2).repeat(res,3)
        if self.isTrain:
            self.params["noise"] = self.params["noise"].repeat(dim,4)


        self.advDiff.density = createParameterizedGrid(self.advDiff.density.data, "scalarSimple", 0, dim, self.params)
        self.advDiff.velocity = createParameterizedGrid(self.advDiff.velocity.data, "vectorComplex", 0, dim, self.params)
        self.advDiff.diffusivity = self.params["nu"]

        self.log["Timestamp"] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.log["Solver"] = {"Resolution": [batch, res, res, res], "Numpy Seed": self.seed}
        self.log["Variation"] = {"Parameter" : self.mode}
        self.log["Stats"] = {}
        for i in range(batch):
            self.log["Stats"]["Velocity%02d" % i] = []
            self.log["Stats"]["Density%02d" % i] = []
        temp = {}
        for key in self.params:
            if (key in modes and not key in self.mode) or key in ["f6", "fd"]:
                temp[key] = self.params[key][0].tolist()
            else:
                temp[key] = self.params[key].tolist()
        self.log["Parameters"] = temp


    def step(self):
        print('\n%s Frame %i' % (self.mode, self.advDiff.age))

        self.advDiff.force = createParameterizedGrid(self.advDiff.force.data, "vectorForcing", self.advDiff.age, dim, self.params)
        #noiseSeed = int(time.time()*1000.0) & 0xffffffff # convert to 32 bit seed
        if self.isTrain:
            #self.advDiff.noise = generateNoise(self.advDiff.noise, self.params["noise"], 0, seed=noiseSeed)
            self.advDiff.noise = Noise(channels=dim, scale=10, smoothness=1.0) * self.params["noise"]
        else:
            #self.advDiff.noiseScalar = generateNoise(self.advDiff.noiseScalar, self.params["noise"], 0, seed=noiseSeed)
            self.advDiff.noiseScalar = Noise(channels=1, scale=10, smoothness=1.0) * self.params["noise"]


        # stats and rendering
        velNP = self.advDiff.velocity.data
        densNP = self.advDiff.density.data
        for i in range(batch):
            self.log["Stats"]["Velocity%02d" % i].append( "Min:%0.8f Max:%0.8f Avg: %0.8f" % (np.min(velNP[i]), np.max(velNP[i]), np.mean(velNP[i])) )
            self.log["Stats"]["Density%02d" % i].append( "Min:%0.8f Max:%0.8f Avg: %0.8f" % (np.min(densNP[i]), np.max(densNP[i]), np.mean(densNP[i])) )
        self.renderData.append( prepareRender(densNP, 4) )

        #if self.advDiff.age > 30:
        if self.advDiff.age == 120:
            np.savez_compressed( self.basepath + "/density_%06d.npz" % self.advDiff.age, self.advDiff.density.data.astype(np.float32) )

        self.advDiff.step()


    def writeLog(self):
        logFile = self.basepath + "/src/descriptionSimple.json"
        with open(logFile, 'w') as f:
            json.dump(self.log, f, indent=4)


    def writeRender(self):
        renderpath = self.basepath + "/render/"

        if not os.path.exists(renderpath):
            os.makedirs(renderpath)

        newName = os.path.join(renderpath, "%s_sim_%06d.mp4" % (self.basename, self.seed))
        imageio.mimwrite(newName, self.renderData, quality=6, fps=11, ffmpeg_log_level="error")

        newNameCopy = os.path.join(destVidCopy, "%s_sim_%06d.mp4" % (self.basename, self.seed))
        imageio.mimwrite(newNameCopy, self.renderData, quality=6, fps=11, ffmpeg_log_level="error")




# parse arguments
if len(sys.argv) == 4:
    mode = modes[int(sys.argv[1])]
    seed = int(sys.argv[2])
    isTrain = sys.argv[3] != "Test"

    print("--------------------------------------------")
    print("| Mode: %s" % mode)
    print("| Seed: %i" % seed)
    print("| Trainset: %s" % isTrain)
    print("--------------------------------------------\n")
else:
    mode = modes[0]
    seed = 0
    isTrain = True

    print("Wrong parameters!")
    exit(1)


# run
if isTrain:
    basename = "advdiff_%s" % mode
    basepath = "%s%s/sim_%06d" % (destTrain, basename, seed)
else:
    basename = "advdiff_dens_%s" % mode
    basepath = "%s%s/sim_%06d" % (destTest, basename, seed)

adv = AdvDiffSim(mode, seed, isTrain, basename, basepath)

if withGUI:
    show(adv, framerate=2)
else:
    adv.prepare()
    for i in range(121):
        adv.progress()

    adv.writeLog()
    adv.writeRender()
