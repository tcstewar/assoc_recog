import nengo
import numpy as np
import nengo.spa as spa
import pytry

class FinishedException(nengo.exceptions.NengoException):
    pass

class Retrieve4Trial(pytry.NengoTrial):
    def params(self):
        self.param('dimensions', D=48)
        self.param('memory size', M=30)
        self.param('recall timeout', timeout=5)
        self.param('minimum time', min_time=0.01)
        self.param('synapse', syn=0.1)
        self.param('input synapse', input_syn=0.005)
        self.param('noise', noise=0.0)
        self.param('zero time', t_zero=0.05)
        self.param('common', common=0.0)

    def model(self, p):
        vocab = spa.Vocabulary(p.D)
        items = []
        fan = []
        assert p.M % 6 == 0

        for i in range(p.M / 6):
            for j in range(3):
                pp = vocab.parse('X%d+Y%d+%g*Q' % (6*i+j, 6*i+j, p.common))
                pp.normalize()
                vocab.add('P%d' % (3*i+j), pp)
                items.append(pp.v)
                fan.append(1)

            for j, v in enumerate(['AB', 'CB', 'AC']):
                pp = vocab.parse('%s%d+%s%d+%g*Q' % (v[0], i, v[1], i, p.common))
                pp.normalize()
                vocab.add('P%d_2' % (3*i+j), pp)
                items.append(pp.v)
                fan.append(2)

        rng = np.random.RandomState(seed=p.seed)
        order = np.arange(len(items))
        rng.shuffle(order)
        items = [items[o] for o in order]
        fan = [fan[o] for o in order]
        self.fan = fan

        model = spa.SPA()
        with model:

            model.cue = spa.State(p.D, vocab=vocab)
            if p.noise > 0:
                for ens in model.cue.all_ensembles:
                    ens.noise = nengo.processes.WhiteNoise(dist=nengo.dists.Gaussian(0, std=p.noise))

            model.mem = nengo.networks.EnsembleArray(n_neurons=50, n_ensembles=len(items))
            for ens in model.mem.all_ensembles:
                ens.encoders = nengo.dists.Choice([[1]])
                ens.intercepts = nengo.dists.Uniform(0, 1)
            nengo.Connection(model.mem.output, model.mem.input,
                             transform=np.eye(len(items))-1,
                             synapse=p.syn)

            nengo.Connection(model.cue.output, model.mem.input,
                             transform=np.array(items),
                             synapse=p.input_syn)

            mem_activity = nengo.Node(None, size_in=1)
            for ens in model.mem.all_ensembles:
               nengo.Connection(ens.neurons, mem_activity,
                                transform=np.ones((1, ens.n_neurons)),
                                synapse=None)
            self.p_activity = nengo.Probe(mem_activity, synapse=0.01)


        class Env(nengo.Node):
            def __init__(self, items):
                self.items = items
                self.index = 0
                self.switch_time = 0
                self.times = []
                self.correct = []
                super(Env, self).__init__(self.func, size_in=len(items), size_out=p.D)
            def func(self, t, x):
                if t > self.switch_time + p.min_time + p.t_zero:
                    values = list(sorted(x))
                    if values[-2]<1e-3 or t > (self.switch_time + p.timeout):
                        self.correct.append(np.argmax(x) == self.index)
                        self.index = (self.index + 1) % len(self.items)
                        self.times.append(t - self.switch_time)
                        self.switch_time = t

                if len(self.times) >= len(self.items) and not p.gui:
                    raise FinishedException()

                if t < self.switch_time + p.t_zero:
                    return np.zeros(p.D)
                else:
                    return self.items[self.index]

        with model:
            self.env = Env(items)
            nengo.Connection(self.env, model.cue.input, synapse=None)
            nengo.Connection(model.mem.output, self.env, synapse=0.005)

        return model

    def evaluate(self, p, sim, plt):
        try:
            while True:
                sim.step()
        except FinishedException:
            pass

        mean_times = {1: [], 2: []}
        for i, f in enumerate(self.fan):
            if self.env.correct[i]:
                mean_times[f].append(self.env.times[i])
        for k, v in mean_times.items():
            mean_times[k] = np.mean(v)

        if plt:
            plt.plot(sim.trange(), sim.data[self.p_activity])


        return dict(times=self.env.times, fan=self.fan,
                    mean_times=mean_times,
                    correct=self.env.correct)




