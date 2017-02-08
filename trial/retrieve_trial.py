import nengo
import nengo.spa as spa
import numpy as np
import pytry
import seaborn

class Retrieve(pytry.NengoTrial):
    def params(self):
        self.param('dimensions', D=8)
        self.param('number of memories', M=3)
        self.param('accumulator strength', accum=0.3)
        self.param('number of tests', n_tests=20)
        self.param('time per test', T_test=0.5)
        self.param('feedback', feedback=0.8)

    def model(self, p):
        vocab = spa.Vocabulary(p.D)
        for i in range(p.M):
            vocab.parse('M%d' % i)

        order = np.arange(p.n_tests)
        np.random.shuffle(order)


        model = spa.SPA()
        with model:
            model.cue = spa.State(p.D, vocab=vocab)
            for ens in model.cue.all_ensembles:
                ens.neuron_type=nengo.Direct()

            model.accum = spa.State(p.D, vocab=vocab, feedback=p.feedback)

            model.recall = spa.AssociativeMemory(vocab,
                                                 wta_output=True,
                                                 threshold_output=True)

            model.recalled = spa.State(p.D, vocab=vocab)
            for ens in model.recalled.all_ensembles:
                ens.neuron_type=nengo.Direct()

            nengo.Connection(model.cue.output, model.accum.input,
                             transform=p.accum)
            nengo.Connection(model.recall.output, model.recalled.input)
            nengo.Connection(model.accum.output, model.recall.input)

            model.same = nengo.Ensemble(n_neurons=100, dimensions=1,
                                        encoders=nengo.dists.Choice([[1]]),
                                        intercepts=nengo.dists.Uniform(0.3,1))

            model.dot = nengo.networks.Product(n_neurons=200, dimensions=p.D)
            nengo.Connection(model.cue.output, model.dot.A)
            nengo.Connection(model.recalled.output, model.dot.B)
            nengo.Connection(model.dot.output, model.same, transform=[[1]*p.D])

            def stim(t):
                index = int(t / p.T_test)
                index2 = order[index % len(order)]
                if index % 2 == 0:
                    return 'X%d' % (index2 % p.M)
                else:
                    return 'M%d' % (index2 % p.M)
            model.input = spa.Input(cue=stim)

            self.p_same = nengo.Probe(model.same, synapse=0.01)
        return model

    def evaluate(self, p, sim, plt):
        sim.run(p.T_test * p.n_tests)

        rts = []
        width = int(p.T_test / p.dt)
        data = sim.data[self.p_same]
        for i in range(p.n_tests):
            if i % 2 == 1:
                d = data[width*i:width*(i+1)]
                good = np.where(d>0.5)[0]
                if len(good) == 0:
                    rt = p.T_test
                else:
                    rt = good[0]*p.dt
                rts.append(rt)



        if plt:
            plt.subplot(2, 1, 1)
            plt.plot(sim.trange(), sim.data[self.p_same])
            plt.subplot(2, 1, 2)
            seaborn.distplot(rts, bins=20)


        return dict(
                rts=rts,
                mean_rt = np.mean(rts),
                )




