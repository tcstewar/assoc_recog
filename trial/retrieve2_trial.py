import nengo
import numpy as np
import nengo.spa as spa
import pytry

class FinishedException(nengo.exceptions.NengoException):
    pass

class Retrieve2Trial(pytry.NengoTrial):
    def params(self):
        self.param('dimensions', D=32)
        self.param('memory size', M=2)
        self.param('synapse', syn=0.1)
        self.param('cue strength', cue_strength=0.5)
        self.param('number of eval points', n_eval_points=500)
        self.param('memory noise', mem_noise=0.5)
        self.param('recall timeout', timeout=5)
        self.param('recall threshold', threshold=1.5)
        self.param('minimum time', min_time=0.05)

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
        pts = []
        target = []
        for i in range(p.n_eval_points):
            pp = items[rng.choice(len(items))]
            t = pp + np.random.normal(size=pp.shape)*p.mem_noise
            pts.append(pp)
            target.append(t)

        model = spa.SPA()
        with model:

            model.cue = spa.State(p.D, vocab=vocab)

            model.mem = spa.State(p.D, subdimensions=p.D, vocab=vocab)
            ens = model.mem.all_ensembles[0]
            nengo.Connection(ens, ens, eval_points=pts, function=target, synapse=p.syn)

            nengo.Connection(model.cue.output, model.mem.input, transform=p.cue_strength)


            match = nengo.networks.Product(n_neurons=200, dimensions=p.D)
            nengo.Connection(model.cue.output, match.A, synapse=None)
            nengo.Connection(model.mem.output, match.B, synapse=None)
            for ens in match.all_ensembles:
                ens.neuron_type = nengo.Direct()

            match_score = nengo.Node(None, size_in=1)
            nengo.Connection(match.output, match_score,
                             transform=np.ones((1, p.D)),
                             synapse=None)

        class Env(nengo.Node):
            def __init__(self, items):
                self.items = items
                self.index = 0
                self.switch_time = 0
                self.times = []
                super(Env, self).__init__(self.func, size_in=1, size_out=p.D)
            def func(self, t, x):
                if t > self.switch_time + p.min_time:
                    if x > p.threshold or t > (self.switch_time + p.timeout):
                        self.index = (self.index + 1) % len(self.items)
                        self.times.append(t - self.switch_time)
                        self.switch_time = t

                if len(self.times) >= len(self.items) and not p.gui:
                    raise FinishedException()

                return self.items[self.index]

        with model:
            self.env = Env(items)
            nengo.Connection(self.env, model.cue.input, synapse=None)
            nengo.Connection(match_score, self.env, synapse=0)

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




