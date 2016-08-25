#nengo
import nengo
import nengo.spa as spa
from nengo_extras.vision import Gabor, Mask

#other
import numpy as np
import inspect, os, sys, time, csv, random
import matplotlib.pyplot as plt
import png
import itertools
import base64
import PIL.Image
import cStringIO




#### SETTINGS #####

nengo_gui_on = __name__ == '__builtin__'
ocl = False #use openCL
full_dims = False #use full dimensions or not

#set path based on gui
if nengo_gui_on:
    if sys.platform == 'darwin':
        cur_path = '/Users/Jelmer/Work/EM/MEG_fan/models/nengo/assoc_recog'
    else:
        cur_path = '/share/volume0/jelmer/MEG_fan/models/nengo/assoc_recog'
else:
    cur_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script path


#set dimensions used by the model
if full_dims:
    D = 256 #for real words need at least 320, probably move op to 512 for full experiment
    Dmid = 128
    Dlow = 48
else: #lower dims
    D = 96
    Dmid = 48
    Dlow = 32



#### HELPER FUNCTIONS ####


#display stimuli in gui, works for 28x90 (two words) and 14x90 (one word)
#t = time, x = vector of pixel values
def display_func(t, x):

    #reshape etc
    if np.size(x) > 14*90:
        input_shape = (1, 28, 90)
    else:
        input_shape = (1,14,90)

    values = x.reshape(input_shape) #back to 2d
    values = values.transpose((1, 2, 0))
    values = (values + 1) / 2 * 255. #0-255
    values = values.astype('uint8') #as ints

    if values.shape[-1] == 1:
        values = values[:, :, 0]

    #make png
    png_rep = PIL.Image.fromarray(values)
    buffer = cStringIO.StringIO()
    png_rep.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue())

    #html for nengo
    display_func._nengo_html_ = '''
           <svg width="100%%" height="100%%" viewbox="0 0 %i %i">
           <image width="100%%" height="100%%"
                  xlink:href="data:image/png;base64,%s"
                  style="image-rendering: auto;">
           </svg>''' % (input_shape[2]*2, input_shape[1]*2, ''.join(img_str))



#load stimuli, subj=0 means a subset of the stims of subject 1 (no long words), works well with lower dims
def load_stims(subj=0):

    #pairs and words in experiment for training model
    global target_pairs #learned word pairs
    global target_words #learned words
    global items #all items in exp (incl. new foils)

    #stimuli for running experiment
    global stims #stimuli in experiment
    global stims_target_rpfoils #target re-paired foils stimuli
    global stims_new_foils #new foil stimuli

    #load files (targets/re-paired foils; short new foils; long new foils)
    #ugly, but this way we can use the original stimulus files
    stims = np.genfromtxt(cur_path + '/stims/S' + str(subj) + 'Other.txt', skip_header=True,
                          dtype=[('probe', np.str_, 8), ('fan', 'int'), ('wordlen', np.str_, 8),
                                 ('item1', np.str_, 8), ('item2', np.str_, 8)], usecols=[1,2,3,4,5])
    stimsNFshort = np.genfromtxt(cur_path + '/stims/S' + str(subj) + 'NFShort.txt', skip_header=True,
                                 dtype=[('probe', np.str_, 8), ('fan', 'int'), ('wordlen', np.str_, 8),
                                        ('item1', np.str_, 8), ('item2', np.str_, 8)], usecols=[1,2,3,4,5])


    if not(subj == 0):
        stimsNFlong = np.genfromtxt(cur_path + '/stims/S' + str(subj) + 'NFLong.txt', skip_header=True,
                                dtype=[('probe', np.str_, 8), ('fan', 'int'), ('wordlen', np.str_, 8),
                                       ('item1', np.str_, 8), ('item2', np.str_, 8)], usecols=[1,2,3,4,5])

    #combine
    if not(subj == 0):
        stims = np.hstack((stims, stimsNFshort, stimsNFlong))
    else:
        stims = np.hstack((stims, stimsNFshort))

    stims = stims.tolist()

    #parse out different categories
    target_pairs = []
    target_words = []
    stims_target_rpfoils = []
    stims_new_foils = []
    items = []

    for i in stims:

        # fill items list with all words
        items.append(i[3])
        items.append(i[4])

        # get target pairs
        if i[0] == 'Target':
            target_pairs.append((i[3],i[4]))
            target_words.append(i[3])
            target_words.append(i[4])

        # make separate lists for targets/rp foils and new foils (for presenting experiment)
        if i[0] != 'NewFoil':
            stims_target_rpfoils.append(i)
        else:
            stims_new_foils.append(i)

    # remove duplicates
    items = np.unique(items).tolist()
    target_words = np.unique(target_words).tolist()



# load images for vision
def load_images():

    global X_train, y_train, y_train_words

    indir = cur_path + '/images/'
    files = os.listdir(indir)
    files2 = []

    #select only images for current item set
    for fn in files:
        if fn[-4:] == '.png' and (fn[:-4] in items):
             files2.append(fn)

    X_train = np.empty(shape=(np.size(files2), 90*14),dtype='float32') #images x pixels matrix
    y_train_words = [] #words represented in X_train
    for i,fn in enumerate(files2):
            y_train_words.append(fn[:-4]) #add word

            #read in image and convert to 0-1 vector
            r = png.Reader(indir + fn)
            r = r.asDirect()
            image_2d = np.vstack(itertools.imap(np.uint8, r[2]))
            image_2d /= 255
            image_1d = image_2d.reshape(1,90*14)
            X_train[i] = image_1d

    #numeric labels for words (could present multiple different examples of words, would get same label)
    y_train = np.asarray(range(0,len(np.unique(y_train_words))))
    X_train = 2 * X_train - 1  # normalize to -1 to 1


#returns pixels of image representing item (ie METAL)
def get_image(item):
    return X_train[y_train_words.index(item)]




#### MODEL FUNCTIONS #####

# performs all steps in model ini
def initialize_model(subj=0):

    load_stims(subj)
    load_images()
    initialize_vocabs()


#initialize vocabs
def initialize_vocabs():

    global vocab_vision #low level visual vocab
    global vocab_concepts #vocab with all concepts
    global vocab_learned_words #vocab with learned words
    global vocab_learned_pairs #vocab with learned pairs
    global vocab_motor #upper motor hierarchy (LEFT, INDEX)
    global vocab_fingers #finger activation (L1, R2)
    global vocab_goal #goal vocab
    global vocab_attend #attention vocab

    global train_targets #vector targets to train X_train on
    global vision_mapping #mapping between visual representations and concepts
    global list_of_pairs #list of pairs in form 'METAL_SPARK'
    global motor_mapping #mapping between higher and lower motor areas


    #low level visual representations
    vocab_vision = nengo.spa.Vocabulary(Dmid,max_similarity=.5)
    for name in y_train_words:
        vocab_vision.parse(name)
    train_targets = vocab_vision.vectors


    #word concepts - has all concepts, including new foils
    vocab_concepts = spa.Vocabulary(D, max_similarity=0.2)
    for i in y_train_words:
        vocab_concepts.parse(i)
    vocab_concepts.parse('ITEM1')
    vocab_concepts.parse('ITEM2')


    #vision-concept mapping between vectors
    vision_mapping = np.zeros((D, Dmid))
    for word in y_train_words:
        vision_mapping += np.outer(vocab_vision.parse(word).v, vocab_concepts.parse(word).v).T

    #vocab with learned words
    vocab_learned_words = vocab_concepts.create_subset(target_words)

    #vocab with learned pairs
    list_of_pairs = []
    for item1, item2 in target_pairs:
        #vocab_learned_pairs.parse('%s*ITEM1 + %s*ITEM2' % (item1, item2)) #think this can go, but let's see
        #vocab_learned_pairs.add('%s_%s' % (item1,item2), vocab_learned_pairs.parse('%s*ITEM1 + %s*ITEM2' % (item1, item2)))
        vocab_concepts.add('%s_%s' % (item1,item2), vocab_concepts.parse('%s*ITEM1 + %s*ITEM2' % (item1, item2))) #add pairs to concepts to use same vectors
        list_of_pairs.append('%s_%s' % (item1,item2)) #keep list of pairs notation
    vocab_learned_pairs = vocab_concepts.create_subset(list_of_pairs) #get only pairs
    #print vocab_learned_pairs.keys

    #motor vocabs, just for sim calcs
    vocab_motor = spa.Vocabulary(Dmid) #different dimension to be sure, upper motor hierarchy
    vocab_motor.parse('LEFT+RIGHT+INDEX+MIDDLE')

    vocab_fingers = spa.Vocabulary(Dlow) #direct finger activation
    vocab_fingers.parse('L1+L2+R1+R2')

    #map higher and lower motor
    motor_mapping = np.zeros((Dlow, Dmid))
    motor_mapping += np.outer(vocab_motor.parse('LEFT+INDEX').v, vocab_fingers.parse('L1').v).T
    motor_mapping += np.outer(vocab_motor.parse('LEFT+MIDDLE').v, vocab_fingers.parse('L2').v).T
    motor_mapping += np.outer(vocab_motor.parse('RIGHT+INDEX').v, vocab_fingers.parse('R1').v).T
    motor_mapping += np.outer(vocab_motor.parse('RIGHT+MIDDLE').v, vocab_fingers.parse('R2').v).T

    #goal vocab
    vocab_goal = spa.Vocabulary(Dlow)
    vocab_goal.parse('DO_TASK')
    vocab_goal.parse('RECOG')
    vocab_goal.parse('RESPOND')
    vocab_goal.parse('END')

    #attend vocab
    vocab_attend = vocab_concepts.create_subset(['ITEM1', 'ITEM2'])



#initialize model
def create_model(trial_info=('Target', 1, 'Short', 'METAL', 'SPARK'), hand='RIGHT',seedin=1):

    #print trial_info
    print '\n\n---- NEW MODEL ----'
    global model

    #word presented in current trial
    item1 = trial_info[3]
    item2 = trial_info[4]

    #returns images of current words
    def present_pair(t):
        im1 = get_image(item1)
        im2 = get_image(item2)
        return np.hstack((im1, im2))

    #returns image 1 <100 ms, otherwise image 2
    def present_item(t):
        if t < .1:
            return get_image(item1)
        else:
            return get_image(item2)

    def present_item2(t,output_attend):

        similarities = [np.dot(output_attend, vocab_attend['ITEM1'].v),
                        np.dot(output_attend, vocab_attend['ITEM2'].v)]
        #print similarities

        ret_ima = np.zeros(1260)
        if similarities[0] > .5:
            ret_ima = get_image(item1)
        elif similarities[1] > .5:
            ret_ima = get_image(item2)

        return ret_ima


    model = spa.SPA(seed=seedin)
    with model:

        #display current stimulus pair (not part of model)
        model.pair_input = nengo.Node(present_pair)
        model.pair_display = nengo.Node(display_func, size_in=model.pair_input.size_out)  # to show input
        nengo.Connection(model.pair_input, model.pair_display, synapse=None)


        # control
        model.control_net = nengo.Network()
        with model.control_net:
            model.attend = spa.State(D, vocab=vocab_attend, feedback=.5)  # vocab_attend
            model.goal = spa.State(D, vocab_goal, feedback=1)  # current goal Dlow
            model.target_hand = spa.State(Dmid, vocab=vocab_motor, feedback=1)


        ### vision ###

        # set up network parameters
        n_vis = X_train.shape[1]  # nr of pixels, dimensions of network
        n_hid = 1000  # nr of gabor encoders/neurons

        # random state to start
        rng = np.random.RandomState(9)
        encoders = Gabor().generate(n_hid, (11, 11), rng=rng)  # gabor encoders, 11x11 apparently, why?
        encoders = Mask((14, 90)).populate(encoders, rng=rng,
                                           flatten=True)  # use them on part of the image

        model.visual_net = nengo.Network()
        with model.visual_net:

            #represent currently attended item
            model.attended_item = nengo.Node(present_item)
            #model.attended_item = nengo.Node(present_item2,size_in=model.attend.output.size_out)
            #nengo.Connection(model.attend.output, model.attended_item)

            model.vision_gabor = nengo.Ensemble(n_hid, n_vis, eval_points=X_train,
                                                    neuron_type=nengo.LIFRate(),
                                                    intercepts=nengo.dists.Choice([-0.5]),
                                                    max_rates=nengo.dists.Choice([100]),
                                                    encoders=encoders)

            model.visual_representation = nengo.Ensemble(n_hid, dimensions=Dmid)

            model.visconn = nengo.Connection(model.vision_gabor, model.visual_representation, synapse=0.005,
                                            eval_points=X_train, function=train_targets,
                                            solver=nengo.solvers.LstsqL2(reg=0.01))
            nengo.Connection(model.attended_item, model.vision_gabor, synapse=None)

            # display attended item
            model.display_attended = nengo.Node(display_func, size_in=model.attended_item.size_out)  # to show input
            nengo.Connection(model.attended_item, model.display_attended, synapse=None)





        # concepts
        model.concepts = spa.AssociativeMemory(vocab_concepts,wta_output=True,wta_inhibit_scale=1)
        nengo.Connection(model.visual_representation, model.concepts.input, transform=vision_mapping)

        # pair representation
        model.vis_pair = spa.State(D, vocab=vocab_concepts, feedback=2)

        model.dm_learned_words = spa.AssociativeMemory(vocab_learned_words) #familiarity should be continuous over all items, so no wta
        nengo.Connection(model.dm_learned_words.output,model.dm_learned_words.input,transform=.5,synapse=.01)

        model.familiarity = spa.State(1,feedback_synapse=.01) #no fb syn specified
        nengo.Connection(model.dm_learned_words.am.elem_output,model.familiarity.input, #am.element_output == all outputs, we sum
                         transform=.8*np.ones((1,model.dm_learned_words.am.elem_output.size_out)))

        model.dm_pairs = spa.AssociativeMemory(vocab_learned_pairs, input_keys=list_of_pairs,wta_output=True)
        nengo.Connection(model.dm_pairs.output,model.dm_pairs.input,transform=.5)

        #this works:
        model.representation = spa.AssociativeMemory(vocab_learned_pairs, input_keys=list_of_pairs, wta_output=True)
        nengo.Connection(model.representation.output, model.representation.input, transform=2)
        model.rep_filled = spa.State(1,feedback_synapse=.005) #no fb syn specified
        nengo.Connection(model.representation.am.elem_output,model.rep_filled.input, #am.element_output == all outputs, we sum
                         transform=.8*np.ones((1,model.representation.am.elem_output.size_out)),synapse=0)

        #this doesn't:
        #model.representation = spa.State(D,feedback=1)
        #model.rep_filled = spa.State(1,feedback_synapse=.005) #no fb syn specified
        #nengo.Connection(model.representation.output,model.rep_filled.input, #am.element_output == all outputs, we sum
        #                 transform=.8*np.ones((1,model.representation.output.size_out)),synapse=0)


        # this shouldn't really be fixed I think
        model.comparison = spa.Compare(D, vocab=vocab_concepts)


        #motor
        model.motor_net = nengo.Network()
        with model.motor_net:

            #input multiplier
            model.motor_input = spa.State(Dmid,vocab=vocab_motor)

            #higher motor area (SMA?)
            model.motor = spa.State(Dmid, vocab=vocab_motor,feedback=1)

            #connect input multiplier with higher motor area
            nengo.Connection(model.motor_input.output,model.motor.input,synapse=.1,transform=10)

            #finger area
            model.fingers = spa.AssociativeMemory(vocab_fingers, input_keys=['L1', 'L2', 'R1', 'R2'], wta_output=True)

            #conncetion between higher order area (hand, finger), to lower area
            nengo.Connection(model.motor.output, model.fingers.input, transform=.4*motor_mapping)

            #finger position (spinal?)
            model.finger_pos = nengo.networks.EnsembleArray(n_neurons=50, n_ensembles=4)
            nengo.Connection(model.finger_pos.output, model.finger_pos.input, synapse=0.1, transform=0.3) #feedback

            #connection between finger area and finger position
            nengo.Connection(model.fingers.am.elem_output, model.finger_pos.input, transform=np.diag([0.55, .53, .57, .55])) #fix these




        model.bg = spa.BasalGanglia(
            spa.Actions(
                'dot(goal,DO_TASK)-.5 --> dm_learned_words=vis_pair, goal=RECOG, attend=ITEM1',
                'dot(goal,RECOG)+dot(attend,ITEM1)+familiarity-2 --> goal=RECOG2, dm_learned_words=vis_pair, attend=ITEM2',#'vis_pair=ITEM1*concepts',
                'dot(goal,RECOG)+dot(attend,ITEM1)+(1-familiarity)-2 --> goal=RECOG2, attend=ITEM2', #motor_input=1.5*target_hand+MIDDLE,
                'dot(goal,RECOG2)+dot(attend,ITEM2)+familiarity-1.3 --> goal=RECOLLECTION,dm_pairs = 2*vis_pair, representation=3*dm_pairs',# vis_pair=ITEM2*concepts',
                'dot(goal,RECOG2)+dot(attend,ITEM2)+(1-familiarity)-1.3 --> goal=RESPOND, motor_input=1.0*target_hand+MIDDLE',
                'dot(goal,RECOLLECTION) - .5 --> goal=RECOLLECTION, representation=2*dm_pairs',
                'dot(goal,RECOLLECTION) + 2*rep_filled - 1.3 --> goal=COMPARE_ITEM1, attend=ITEM1, comparison_A = 2*vis_pair,comparison_B = 2*representation*~attend',
                'dot(goal,COMPARE_ITEM1) + rep_filled + comparison -1 --> goal=COMPARE_ITEM2, attend=ITEM2, comparison_A = 2*vis_pair',#comparison_B = 2*representation*~attend',
                'dot(goal,COMPARE_ITEM1) + rep_filled + (1-comparison) -1 --> goal=RESPOND,motor_input=1.0*target_hand+MIDDLE',#comparison_A = 2*vis_pair,comparison_B = 2*representation*~attend',
                'dot(goal,COMPARE_ITEM2) + rep_filled + comparison - 1 --> goal=RESPOND,motor_input=1.0*target_hand+INDEX',#comparison_A = 2*vis_pair,comparison_B = 2*representation*~attend',
                'dot(goal,COMPARE_ITEM2) + rep_filled + (1-comparison) -1 --> goal=RESPOND,motor_input=1.0*target_hand+MIDDLE',#comparison_A = 2*vis_pair,comparison_B = 2*representation*~attend',

                'dot(goal,RESPOND) + comparison - 1 --> goal=RESPOND, motor_input=1.0*target_hand+INDEX', #comparison_A = 2*vis_pair,comparison_B = 2*representation*~attend',
                'dot(goal,RESPOND) + (1-comparison) - 1 --> goal=RESPOND, motor_input=1.0*target_hand+MIDDLE', #comparison_A = 2*vis_pair,comparison_B = 2*representation*~attend',

                # 'dot(goal,RECOLLECTION) + (1 - dot(representation,vis_pair)) - 1.3 --> goal=RESPOND, motor_input=1.0*target_hand+MIDDLE',
                'dot(goal,RESPOND)+dot(motor,MIDDLE+INDEX)-1.0 --> goal=END',
                'dot(goal,END) --> goal=END',
                #'.6 -->',

                #possible to match complete buffer, ie is representation filled?

            ))
        model.thalamus = spa.Thalamus(model.bg)

        model.cortical = spa.Cortical( # cortical connection: shorthand for doing everything with states and connections
            spa.Actions(
              #  'motor_input = .04*target_hand',
                #'dm_learned_words = .8*concepts', #.5
                #'dm_pairs = 2*stimulus'
                'vis_pair = 2*attend*concepts+concepts',
                'comparison_A = 2*vis_pair',
                'comparison_B = 2*representation*~attend',

            ))


        #probes
        #model.pr_goal = nengo.Probe(model.goal.output,synapse=.01)
        model.pr_motor_pos = nengo.Probe(model.finger_pos.output,synapse=.01) #raw vector (dimensions x time)
        model.pr_motor = nengo.Probe(model.fingers.output,synapse=.01)
        model.pr_motor1 = nengo.Probe(model.motor.output, synapse=.01)
        #model.pr_target = nengo.Probe(model.target_hand.output, synapse=.01)
        #model.pr_attend = nengo.Probe(model.attend.output, synapse=.01)

        #input
        model.input = spa.Input(goal=lambda t: 'DO_TASK' if t < 0.05 else '0',
                                target_hand=hand,
                                #attend=lambda t: 'ITEM1' if t < 0.1 else 'ITEM2',
                                )

        #print(sum(ens.n_neurons for ens in model.all_ensembles))

        #return model
        ### END MODEL




##### EXPERIMENTAL CONTROL #####

results = []

#save results to file
def save_results(fname='output'):
    with open(cur_path + '/data/' + fname + '.txt', "w") as f:
        writer = csv.writer(f)
        writer.writerows(results)


#prepare simulation
def prepare_sim():

    global sim

    print('\nModel preparation, ' + str(D) + ' dimensions, ' +
          str(sum(ens.n_neurons for ens in model.all_ensembles))
          + ' neurons, and ' + str(len(vocab_concepts.keys)) + ' concepts...')

    start = time.clock()

    if ocl:
        sim = nengo_ocl.Simulator(model)
    else:
        sim = nengo.Simulator(model)
    print('\t\t\t...finished in ' + str(round(time.clock() - start,2)) + ' seconds.\n')



#trial_info = target/foil, fan, word length, item 1, item 2
total_sim_time = 0

def do_trial(trial_info=('Target', 1, 'Short', 'METAL', 'SPARK'),hand='RIGHT'):

    global total_sim_time
    global results

    print('\nStart trial: ' + trial_info[0] + ', Fan ' + str(trial_info[1])
          + ', ' + trial_info[2] + ' - ' + ' '.join(trial_info[3:]) + ' - ' + hand + '\n')

    #run sim at least 100 ms
    sim.run(.1) #make this shorter than fastest RT

    print('Stepped sim started...\n')

    stepsize = 5 #ms
    resp = -1
    while sim.time < 1:

        # run stepsize ms, update time
        sim.run_steps(stepsize, progress_bar=False)

        #target_h = sim.data[model.pr_target][sim.n_steps-1]
        #print np.dot(target_h, vocab_motor['RIGHT'].v)

        #calc finger position
        last_motor_pos = sim.data[model.pr_motor_pos][sim.n_steps-1 ]
        position_finger = np.max(last_motor_pos)

        if position_finger >.68: #.68 represents key press
            break


    # determine response
    step = sim.n_steps
    similarities = [np.dot(sim.data[model.pr_motor][step - 1], vocab_fingers['L1'].v),
                    np.dot(sim.data[model.pr_motor][step - 1], vocab_fingers['L2'].v),
                    np.dot(sim.data[model.pr_motor][step - 1], vocab_fingers['R1'].v),
                    np.dot(sim.data[model.pr_motor][step - 1], vocab_fingers['R2'].v)]
    resp = np.argmax(similarities)
    if resp == 0:
        print '\nLeft Index'
    elif resp == 1:
        print '\nLeft Middle'
    elif resp == 2:
        print '\nRight Index'
    elif resp == 3:
        print '\nRight Middle'
    if resp == -1:
        print '\nNo response'
    print('\n... and done!')


    #resp 0 = left index, 1 = left middle, 2 = right index, 3 = right middle
    #change coding later
    if (trial_info[0] == 'Target' or trial_info[0] == 'RPFoil') and ((resp == 0 and hand == 'LEFT') or (resp == 2 and hand == 'RIGHT')):
        acc = 1
    elif (trial_info[0] == 'NewFoil') and ((resp == 1 and hand == 'LEFT') or (resp == 3 and hand == 'RIGHT')):
        acc = 1
    else:
        acc = 0

    print('\nRT = ' + str(sim.time) + ', acc = ' + str(acc))
    total_sim_time += sim.time
    results.append(trial_info + (hand, np.round(sim.time,3), acc, resp))



def do_1_trial(trial_info=('Target', 1, 'Short', 'METAL', 'SPARK'),hand='RIGHT'):

    global total_sim_time
    global results
    total_sim_time = 0
    results = []

    start = time.clock()
    create_model(trial_info,hand)
    prepare_sim()
    do_trial(trial_info,hand)

    print('\nTotal time: ' + str(round(time.clock() - start,2)) + ' seconds for ' + str(total_sim_time) + ' seconds simulation.')
    sim.close()

    print('\n')
    print(results)



def do_4_trials():

    global results
    total_sim_time = 0
    results = []

    start = time.clock()

    stims_in = []
    for i in [0,33, 32,1]:
        stims_in.append(stims[i])

    hands_in = ['RIGHT','LEFT','RIGHT','LEFT']

    for i in range(4):
        cur_trial = stims_in[i]
        cur_hand = hands_in[i]
        create_model(cur_trial, cur_hand)
        prepare_sim()
        do_trial(cur_trial, cur_hand)
        sim.close()

    print(
    '\nTotal time: ' + str(round(time.clock() - start, 2)) + ' seconds for ' + str(total_sim_time) + ' seconds simulation.')

    # save behavioral data
    print('\n')
    print results
    save_results()




def do_1_block(cur_hand='RIGHT'):

    global results
    results = []
    total_sim_time = 0
    start = time.clock()

    stims_in = stims_target_rpfoils
    nr_trp = len(stims_target_rpfoils)
    nr_nf = nr_trp / 4

    #add new foils
    stims_in = stims_in + random.sample(stims_new_foils, nr_nf)

    #shuffle
    random.shuffle(stims_in)

    for i in stims_in:
        cur_trial = i
        create_model(cur_trial, cur_hand)
        prepare_sim()
        do_trial(cur_trial, cur_hand)
        sim.close()

    print(
    '\nTotal time: ' + str(round(time.clock() - start, 2)) + ' seconds for ' + str(total_sim_time) + ' seconds simulation.')


    # save behavioral data
    print('\n')
    save_results('output' + '_' + cur_hand + '.txt')



#choice of trial, etc
if not nengo_gui_on:
    print 'nengo gui not on'

    full_dims = False
    initialize_model(subj=0)


    do_1_trial(trial_info=('Target', 1, 'Short', 'METAL', 'SPARK'), hand='RIGHT')
    #do_1_trial(trial_info=('NewFoil', 1, 'Short', 'CARGO', 'HOOD'),hand='LEFT')
    #do_1_trial(trial_info=('RPFoil', 1,	'Short', 'SODA', 'BRAIN'), hand='RIGHT')

    #do_4_trials()
    #do_1_block('RIGHT')
    #do_1_block('LEFT')

else:

    print 'nengo gui on'
    initialize_model(subj=0)
    create_model()



