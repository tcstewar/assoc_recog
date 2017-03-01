import nengo
import numpy as np
import nengo.spa as spa

D = 32
M = 2
seed = 1
syn = 0.1
cue_strength = 0.5
vocab = spa.Vocabulary(D)
items = []
for i in range(M):
    p = vocab.parse('X%d+Y%d' % (i, i))
    p.normalize()
    vocab.add('P%d' % i, p)
    items.append(p.v)
for i in range(M):
    if i % 2 == 0:
        p = vocab.parse('X%d+Z%d' % (i, i))
        p.normalize()
        vocab.add('P%d_2' % i, p)
        items.append(p.v)
        
    
rng = np.random.RandomState(seed=seed)
n_eval_points = 500
pts = []
target = []
mem_noise = 0.5
for i in range(n_eval_points):
    p = items[rng.choice(len(items))]
    t = p + np.random.normal(size=p.shape)*mem_noise
    pts.append(p)
    target.append(t)

model = spa.SPA()
with model:
    
    model.cue = spa.State(D, vocab=vocab)
    
    model.mem = spa.State(D, subdimensions=D, vocab=vocab)
    ens = model.mem.all_ensembles[0]
    nengo.Connection(ens, ens, eval_points=pts, function=target, synapse=syn)

    nengo.Connection(model.cue.output, model.mem.input, transform=cue_strength)
    
    
    match = nengo.networks.Product(n_neurons=200, dimensions=D)
    nengo.Connection(model.cue.output, match.A)
    nengo.Connection(model.mem.output, match.B)
    for ens in match.all_ensembles:
        ens.neuron_type = nengo.Direct()
    
    match_score = nengo.Node(None, size_in=1)
    nengo.Connection(match.output, match_score, transform=np.ones((1, D)))
    
    