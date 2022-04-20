import os, sys, time, json, datetime
import numpy as np
import imageio

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from phi.tf.flow import *
from varied_sim_utils import *


withGUI = False
modes = ["f1", "f1neg", "f2", "f2neg", "f3", "f3neg", "f4", "f4neg", "f5", "f5neg", "f7", "f7neg",
        "o1", "o1neg", "o2", "o2neg", "noise",] # count: 17

# solver params
res = 128
batch = 11
dim = 3
dest = "data/train/"
destVidCopy = "data/videos/"


@struct.definition()
class BurgersForcing(DomainState):

    def __init__(self, domain, velocity=0, force=0, noise=0, diffusivity=0, tags=('burgers', 'velocityfield'),
                    name='burgers', **kwargs):
        DomainState.__init__(self, **struct.kwargs(locals()))

    @struct.variable(default=0.0, dependencies=DomainState.domain)
    def velocity(self, velocity):
        return self.centered_grid('velocity', velocity, self.rank)

    @struct.variable(default=0.0, dependencies=DomainState.domain)
    def noise(self, noise):
        return self.centered_grid('noise', noise, self.rank)

    @struct.variable(default=0.0, dependencies=DomainState.domain)
    def force(self, force):
        return self.centered_grid('force', force, self.rank)


    @struct.constant(default=0.0)
    def diffusivity(self, diffusivity):
        return diffusivity


class BurgersForcingPhysics(Physics):

    def __init__(self):
        Physics.__init__(self, [StateDependency('effects', 'velocity_effect', blocking=True)])

    def step(self, state, dt=1.0, effects=()):
        nu = state.diffusivity

        vel = state.velocity
        force = state.force
        noise = state.noise

        vel = advect.semi_lagrangian(vel, vel, dt)
        vel = diffuse(vel, dt * nu, substeps=1)

        vel = vel + force
        vel = vel + noise

        for effect in effects:
            vel = effect_applied(effect, vel, dt)

        return state.copied_with(velocity=vel, age=state.age + 1)


class BurgersSim(App):

    def __init__(self, mode, seed, basename, basepath):
        App.__init__(self, basename, base_dir=dest)
        self.mode = mode
        self.seed = seed
        self.basename = basename
        self.basepath = basepath
        self.renderData = []
        self.log = {}

        state = BurgersForcing( Domain((res, res, res), boundaries=PERIODIC), batch_size=batch,
                                velocity=Noise(channels=dim)*0, force=Noise(channels=dim)*0, noise=Noise(channels=dim)*0 )
        self.burgers = world.add(state, physics=BurgersForcingPhysics())
        self.setupSim()

        self.add_field('Velocity', lambda: self.burgers.velocity)
        self.add_field('Force', lambda: self.burgers.force)
        self.add_field('Noise', lambda: self.burgers.noise)

    def action_reset(self):
        self.setupSim()
        self.steps = 0
        self.burgers.age = 0

    def setupSim(self):
        #np.random.seed(self.seed)
        self.params = generateParams(self.mode, batch, dim, 0.060,
                                    0.014, 0.005, 0.007, 0.015, 0.030, 0.300,
                                    0.06, 0.12, 0.0, 0.01)
        self.params["noise"] = self.params["noise"][:,None,None,None,None].repeat(res,1).repeat(res,2).repeat(res,3).repeat(dim,4)


        self.burgers.velocity = createParameterizedGrid(self.burgers.velocity.data, "vectorComplex", 0, dim, self.params)
        self.burgers.diffusivity = self.params["nu"]*0.1

        self.log["Timestamp"] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.log["Solver"] = {"Resolution": [batch, res, res, res], "Numpy Seed": self.seed}
        self.log["Variation"] = {"Parameter" : self.mode}
        self.log["Stats"] = {}
        for i in range(batch):
            self.log["Stats"]["Velocity%02d" % i] = []
        temp = {}
        for key in self.params:
            if (key in modes and not key in self.mode) or key in ["f6"]:
                temp[key] = self.params[key][0].tolist()
            else:
                temp[key] = self.params[key].tolist()
        self.log["Parameters"] = temp


    def step(self):
        print('\n%s Frame %i' % (self.mode, self.burgers.age))

        self.burgers.force = createParameterizedGrid(self.burgers.force.data, "vectorForcing", self.burgers.age, dim, self.params)
        #noiseSeed = int(time.time()*1000.0) & 0xffffffff # convert to 32 bit seed
        #self.burgers.noise = generateNoise(self.burgers.noise, self.params["noise"], 0, seed=noiseSeed)
        self.burgers.noise = Noise(channels=dim, scale=10, smoothness=1.0) * self.params["noise"]

        # stats and rendering
        velNP = self.burgers.velocity.data
        for i in range(batch):
            self.log["Stats"]["Velocity%02d" % i].append( "Min:%0.8f Max:%0.8f Avg: %0.8f" % (np.min(velNP[i]), np.max(velNP[i]), np.mean(velNP[i])) )
        self.renderData.append( prepareRender(velNP, 4) )

        #if self.burgers.age > 30:
        if self.burgers.age == 120:
            np.savez_compressed( self.basepath + "/velocity_%06d.npz" % self.burgers.age, self.burgers.velocity.data.astype(np.float32) )

        self.burgers.step()


    def writeLog(self):
        logFile = self.basepath + "/src/descriptionSimple.json"
        with open(logFile, 'w') as f:
            json.dump(self.log, f, indent=4)


    def writeRender(self):
        renderpath = basepath + "/render/"

        if not os.path.exists(renderpath):
            os.makedirs(renderpath)

        newName = os.path.join(renderpath, "%s_sim_%06d.mp4" % (self.basename, self.seed))
        imageio.mimwrite(newName, self.renderData, quality=6, fps=11, ffmpeg_log_level="error")

        newNameCopy = os.path.join(destVidCopy, "%s_sim_%06d.mp4" % (self.basename, self.seed))
        imageio.mimwrite(newNameCopy, self.renderData, quality=6, fps=11, ffmpeg_log_level="error")




# parse arguments
if len(sys.argv) == 3:
    mode = modes[int(sys.argv[1])]
    seed = int(sys.argv[2])

    print("--------------------------------------------")
    print("| Mode: %s" % mode)
    print("| Seed: %i" % seed)
    print("--------------------------------------------\n")
else:
    mode = modes[0]
    seed = 0

    print("Wrong parameters!")
    #exit(1)

# run
basename = "burgers_%s" % mode
basepath = "%s%s/sim_%06d" % (dest, basename, seed)

bur = BurgersSim(mode, seed, basename, basepath)

if withGUI:
    show(bur, framerate=2)
else:
    bur.prepare()
    for i in range(121):
        bur.progress()

    bur.writeLog()
    bur.writeRender()