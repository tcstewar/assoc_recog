import nengo
import numpy as np
import nengo.spa as spa
import pytry

class FinishedException(nengo.exceptions.NengoException):
    pass

class Retrieve2Trial(pytry.NengoTrial):
    def params(self):
        self.param('dimensions', D=48)
        self.param('memory size', M=32)
        self.param('recall timeout', timeout=5)
        self.param('minimum time', min_time=0.01)
        self.param('synapse', syn=0.1)

    def model(self, p):
        vocab = spa.Vocabulary(p.D)
        items = []
        total_fan = []
        for i in range(p.M):
            pp = vocab.parse('X%d+Y%d' % (i, i))
            pp.normalize()
            vocab.add('P%d' % i, pp)
            items.append(pp.v)

            if i % 2 == 0:
                total_fan.append(3)
                pp = vocab.parse('X%d+Z%d' % (i, i))
                pp.normalize()
                vocab.add('P%d_2' % i, pp)
                items.append(pp.v)
                total_fan.append(3)
            else:
                total_fan.append(2)




        rng = np.random.RandomState(seed=p.seed)
        order = np.arange(len(items))
        rng.shuffle(order)
        items = [items[o] for o in order]
        total_fan = [total_fan[o] for o in order]
        self.total_fan = total_fan

        model = spa.SPA()
        with model:

            model.cue = spa.State(p.D, vocab=vocab)

            model.mem = nengo.networks.EnsembleArray(n_neurons=50, n_ensembles=len(items))
            for ens in model.mem.all_ensembles:
                ens.encoders = nengo.dists.Choice([[1]])
                ens.intercepts = nengo.dists.Uniform(0, 1)
            nengo.Connection(model.mem.output, model.mem.input,
                             transform=np.eye(len(items))-1,
                             synapse=p.syn)

            nengo.Connection(model.cue.output, model.mem.input, transform=np.array(items))

        class Env(nengo.Node):
            def __init__(self, items):
                self.items = items
                self.index = 0
                self.switch_time = 0
                self.times = []
                super(Env, self).__init__(self.func, size_in=len(items), size_out=p.D)
            def func(self, t, x):
                if t > self.switch_time + p.min_time:
                    values = list(sorted(x))
                    if values[-2]<1e-3 or t > (self.switch_time + p.timeout):
                        self.index = (self.index + 1) % len(self.items)
                        self.times.append(t - self.switch_time)
                        self.switch_time = t

                if len(self.times) >= len(self.items) and not p.gui:
                    raise FinishedException()

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

        mean_times = {2: [], 3: []}
        for i, f in enumerate(self.total_fan):
            mean_times[f].append(self.env.times[i])
        for k, v in mean_times.items():
            mean_times[k] = np.mean(v)


        return dict(times=self.env.times, total_fan=self.total_fan,
                    mean_times=mean_times)




