import numpy as np
import matplotlib.pyplot as plt
import nengo
import nengo_ocl
import os, inspect, sys

ocl = True #use openCL
nengo_gui_on = __name__ == '__builtin__'


#set path based on gui
if nengo_gui_on:
    if sys.platform == 'darwin':
        cur_path = '/Users/Jelmer/Work/EM/MEG_fan/models/nengo/assoc_recog'
    else:
        cur_path = '/share/volume0/jelmer/MEG_fan/models/nengo/assoc_recog'
else:
    cur_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script path


if sys.platform == 'darwin':
    os.environ["PYOPENCL_CTX"] = "0:1"
else:
    os.environ["PYOPENCL_CTX"] = "0"
	

# define the model
with nengo.Network() as model:
    stim = nengo.Node(np.sin)
    a = nengo.Ensemble(100, 1)
    b = nengo.Ensemble(100, 1)
    nengo.Connection(stim, a)
    nengo.Connection(a, b, function=lambda x: x**2)

    probe_a = nengo.Probe(a, synapse=0.01)
    probe_b = nengo.Probe(b, synapse=0.01)

# build and run the model
with nengo_ocl.Simulator(model) as sim:
#with nengo.Simulator(model) as sim:
    sim.run(10)

# plot the results
plt.plot(sim.trange(), sim.data[probe_a])
plt.plot(sim.trange(), sim.data[probe_b])
plt.show()
