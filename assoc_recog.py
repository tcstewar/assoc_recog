#nengo
import nengo
import nengo.spa as spa
from nengo_extras.vision import Gabor, Mask

#other
import numpy as np
import inspect, os, sys, time, csv, random
import matplotlib.pyplot as plt
import png ##pypng
import itertools
import base64
import PIL.Image
import cStringIO
import socket
import warnings

#open cl settings
if sys.platform == 'darwin':
    os.environ["PYOPENCL_CTX"] = "0:1"
elif socket.gethostname() == 'ai17864':
	print('ai comp')
else:
    os.environ["PYOPENCL_CTX"] = "0"
	


#### SETTINGS #####

nengo_gui_on = __name__ == '__builtin__'
ocl = True #use openCL
high_dims = False #use full dimensions or not
verbose = True

print('\nSettings:')

if ocl:
	print('\tOpenCL ON')
	import pyopencl
	import nengo_ocl
	ctx = pyopencl.create_some_context()
else:
	print('\tOpenCL OFF')


#set path based on gui
if nengo_gui_on:
    print('\tNengo GUI ON')
    if sys.platform == 'darwin':
        cur_path = '/Users/Jelmer/Work/EM/MEG_fan/models/nengo/assoc_recog'
    elif socket.gethostname() == 'ai17864':
    	cur_path = '/home/p234584/assoc_recog'
    else:
        cur_path = '/share/volume0/jelmer/MEG_fan/models/nengo/assoc_recog'
else:
    print('\tNengo GUI OFF')
    cur_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script path


#set dimensions used by the model
if high_dims:
    D = 256 #for real words need at least 320, probably move op to 512 for full experiment
    Dmid = 128
    Dlow = 48
    print('\tFull dimensions: D = ' + str(D) + ', Dmid = ' + str(Dmid) + ', Dlow = ' + str(Dlow))
else: #lower dims
    D = 96
    Dmid = 48
    Dlow = 32
    print('\tLow dimensions: D = ' + str(D) + ', Dmid = ' + str(Dmid) + ', Dlow = ' + str(Dlow))

print('')


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
    global rpfoil_pairs #presented rp_foils
    global newfoil_pairs #presented new foils

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
    rpfoil_pairs = []
    newfoil_pairs = []
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
        elif i[0] == 'RPFoil':
            rpfoil_pairs.append((i[3],i[4]))
        else:
            newfoil_pairs.append((i[3],i[4]))


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

    #warn when loading full stim set with low dimensions:
    if subj > 0 and not(high_dims):
        warn = warnings.warn('Initializing model with full stimulus set, but using low dimensions for vocabs.')

    load_stims(subj)
    load_images()
    initialize_vocabs()


#initialize vocabs
def initialize_vocabs():

    print '---- INITIALIZING VOCABS ----'

    global vocab_vision #low level visual vocab
    global vocab_concepts #vocab with all concepts
    global vocab_learned_words #vocab with learned words
    global vocab_learned_pairs #vocab with learned pairs
    global vocab_motor #upper motor hierarchy (LEFT, INDEX)
    global vocab_fingers #finger activation (L1, R2)
    global vocab_goal #goal vocab
    global vocab_attend #attention vocab
    global vocab_reset #reset vocab

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
    vocab_concepts.parse('NONE')


    #vision-concept mapping between vectors
    vision_mapping = np.zeros((D, Dmid))
    for word in y_train_words:
        vision_mapping += np.outer(vocab_vision.parse(word).v, vocab_concepts.parse(word).v).T

    #vocab with learned words
    vocab_learned_words = vocab_concepts.create_subset(target_words + ['NONE'])


    #vocab with learned pairs
    list_of_pairs = []
    for item1, item2 in target_pairs:
        #vocab_learned_pairs.parse('%s*ITEM1 + %s*ITEM2' % (item1, item2)) #think this can go, but let's see
        #vocab_learned_pairs.add('%s_%s' % (item1,item2), vocab_learned_pairs.parse('%s*ITEM1 + %s*ITEM2' % (item1, item2)))
        vocab_concepts.add('%s_%s' % (item1,item2), vocab_concepts.parse('%s*ITEM1 + %s*ITEM2' % (item1, item2))) #add pairs to concepts to use same vectors
        list_of_pairs.append('%s_%s' % (item1,item2)) #keep list of pairs notation
    vocab_learned_pairs = vocab_concepts.create_subset(list_of_pairs) #get only pairs
    #print vocab_learned_pairs.keys

    #remove all pairs from vis pair?
    #vocab_concepts = vocab_concepts.create_subset(y_train_words + ['ITEM1', 'ITEM2'])



    #add all presented pairs to concepts for display
    for item1, item2 in newfoil_pairs:
        vocab_concepts.add('%s_%s' % (item1, item2), vocab_concepts.parse(
            '%s*ITEM1 + %s*ITEM2' % (item1, item2)))  # add pairs to concepts to use same vectors
    for item1, item2 in rpfoil_pairs:
        vocab_concepts.add('%s_%s' % (item1, item2), vocab_concepts.parse(
            '%s*ITEM1 + %s*ITEM2' % (item1, item2)))  # add pairs to concepts to use same vectors


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

    #reset vocab
    vocab_reset = spa.Vocabulary(Dlow)
    vocab_reset.parse('CLEAR+GO')



# word presented in current trial
global cur_item1
global cur_item2
global cur_hand
cur_item1 = 'METAL' #just for ini
cur_item2 = 'SPARK'
cur_hand = 'LEFT'

# returns images of current words for display
def present_pair(t):
    im1 = get_image(cur_item1)
    im2 = get_image(cur_item2)
    return np.hstack((im1, im2))

# returns image 1 <100 ms, otherwise image 2 || NOT USED ANYMORE
# returns image 1 <100 ms, otherwise image 2 || NOT USED ANYMORE
def present_item(t):
    if t < .1:
        #print(cur_item1)
        return get_image(cur_item1)
    else:
        #print(cur_item2)
        return get_image(cur_item2)



def present_item2(t, output_attend):
    attn = vocab_attend.dot(output_attend) #dot product with current input (ie ITEM 1 or 2)
    i = np.argmax(attn) #index of max

    ret_ima = np.zeros(1260)

    if attn[i] > 0.4: #if we really attend something
        if i == 0: #first item
            ret_ima = get_image(cur_item1)
        else:
            ret_ima = get_image(cur_item2)

    return ret_ima


#get vector representing hand
def get_hand(t):
    #print(cur_hand)
    return vocab_motor.vectors[vocab_motor.keys.index(cur_hand)]



#initialize model
def create_model():

    #print trial_info
    print '---- INTIALIZING MODEL ----'
    global model

    model = spa.SPA()
    with model:

        #display current stimulus pair (not part of model)
        if nengo_gui_on:
            model.pair_input = nengo.Node(present_pair)
            model.pair_display = nengo.Node(display_func, size_in=model.pair_input.size_out)  # to show input
            nengo.Connection(model.pair_input, model.pair_display, synapse=None)


        # control
        model.control_net = nengo.Network()
        with model.control_net:
            #assuming the model knows which hand to use (which was blocked)
            model.hand_input = nengo.Node(get_hand)
            model.target_hand = spa.State(Dmid, vocab=vocab_motor, feedback=1)
            nengo.Connection(model.hand_input,model.target_hand.input,synapse=None)

            model.attend = spa.State(D, vocab=vocab_attend, feedback=.5)  # vocab_attend
            model.goal = spa.State(D, vocab_goal, feedback=1)  # current goal


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
            model.attended_item = nengo.Node(present_item2,size_in=D)
            nengo.Connection(model.attend.output, model.attended_item)

            model.vision_gabor = nengo.Ensemble(n_hid, n_vis, eval_points=X_train,
                                                    neuron_type=nengo.LIFRate(),
                                                    intercepts=nengo.dists.Choice([-0.5]),
                                                    max_rates=nengo.dists.Choice([100]),
                                                    encoders=encoders)

            model.visual_representation = nengo.Ensemble(n_hid, dimensions=Dmid)

            model.visconn = nengo.Connection(model.vision_gabor, model.visual_representation, synapse=0.01, #was .005
                                            eval_points=X_train, function=train_targets,
                                            solver=nengo.solvers.LstsqL2(reg=0.01))
            nengo.Connection(model.attended_item, model.vision_gabor, synapse=None) #synapse?

            # display attended item, only in gui
            if nengo_gui_on:
                model.display_attended = nengo.Node(display_func, size_in=model.attended_item.size_out)  # to show input
                nengo.Connection(model.attended_item, model.display_attended, synapse=None)




        ### central cognition ###

        # concepts
        model.concepts = spa.AssociativeMemory(vocab_concepts,
                                               wta_output=True,
                                               wta_inhibit_scale=1, #was 1
                                               default_output_key='NONE', #what to say if input doesn't match
                                               threshold=0.3)  # how strong does input need to be for it to recognize
        nengo.Connection(model.visual_representation, model.concepts.input, transform=.8*vision_mapping) #not too fast to concepts, might have to be increased to have model react faster to first word.

        #concepts accumulator
        model.concepts_evidence = spa.State(1, feedback=1, feedback_synapse=0.03) #the lower the synapse, the faster it accumulates (was .1)
        concepts_evidence_scale = 2.5
        nengo.Connection(model.concepts.am.elem_output, model.concepts_evidence.input,
                         transform=concepts_evidence_scale * np.ones((1, model.concepts.am.elem_output.size_out)),synapse=0.005)

        #reset if concepts is NONE (default)
        nengo.Connection(model.concepts.am.ensembles[-1], model.concepts_evidence.all_ensembles[0].neurons,
                         transform=np.ones((model.concepts_evidence.all_ensembles[0].n_neurons, 1)) * -40, # was -10
                         synapse=0.005) #lower synapse gives shorter impact of reset - makes the reaction a little slower


        # pair representation
        model.vis_pair = spa.State(D, vocab=vocab_concepts, feedback=1.4) #was 2, 1.6 works ok, but everything gets activated.

        model.dm_learned_words = spa.AssociativeMemory(vocab_learned_words,default_output_key='NONE',threshold=.3) #familiarity should be continuous over all items, so no wta
        nengo.Connection(model.dm_learned_words.output,model.dm_learned_words.input,transform=.4,synapse=.01)


        # this stores the accumulated evidence for or against familiarity
        model.familiarity = spa.State(1, feedback=1, feedback_synapse=0.1) #fb syn influences speed of acc
        familiarity_scale = 0.2
        nengo.Connection(model.dm_learned_words.am.ensembles[-1], model.familiarity.input, transform=-familiarity_scale) #accumulate to -1
        nengo.Connection(model.dm_learned_words.am.elem_output, model.familiarity.input, #am.element_output == all outputs, we sum
                         transform=familiarity_scale * np.ones((1, model.dm_learned_words.am.elem_output.size_out))) #accumulate to 1

        model.do_fam = spa.AssociativeMemory(vocab_reset, default_output_key='CLEAR', threshold=.2)
        nengo.Connection(model.do_fam.am.ensembles[-1], model.familiarity.all_ensembles[0].neurons,
                         transform=np.ones((model.familiarity.all_ensembles[0].n_neurons, 1)) * -10,
                         synapse=0.005)



        #fam model.dm_pairs = spa.AssociativeMemory(vocab_learned_pairs, input_keys=list_of_pairs,wta_output=True)
        #fam nengo.Connection(model.dm_pairs.output,model.dm_pairs.input,transform=.5)

        #this works:
        #fam model.representation = spa.AssociativeMemory(vocab_learned_pairs, input_keys=list_of_pairs, wta_output=True)
        #fam nengo.Connection(model.representation.output, model.representation.input, transform=2)
        #fam model.rep_filled = spa.State(1,feedback_synapse=.005) #no fb syn specified
        #fam nengo.Connection(model.representation.am.elem_output,model.rep_filled.input, #am.element_output == all outputs, we sum
        #fam                  transform=.8*np.ones((1,model.representation.am.elem_output.size_out)),synapse=0)

        #this doesn't:
        #model.representation = spa.State(D,feedback=1)
        #model.rep_filled = spa.State(1,feedback_synapse=.005) #no fb syn specified
        #nengo.Connection(model.representation.output,model.rep_filled.input, #am.element_output == all outputs, we sum
        #                 transform=.8*np.ones((1,model.representation.output.size_out)),synapse=0)


        # this shouldn't really be fixed I think
        #fam model.comparison = spa.Compare(D, vocab=vocab_concepts)


        #motor
        model.motor_net = nengo.Network()
        with model.motor_net:

            #input multiplier
            model.motor_input = spa.State(Dmid,vocab=vocab_motor)

            #higher motor area (SMA?)
            model.motor = spa.State(Dmid, vocab=vocab_motor,feedback=.7)

            #connect input multiplier with higher motor area
            nengo.Connection(model.motor_input.output,model.motor.input,synapse=.1,transform=2)

            #finger area
            model.fingers = spa.AssociativeMemory(vocab_fingers, input_keys=['L1', 'L2', 'R1', 'R2'], wta_output=True)

            #conncetion between higher order area (hand, finger), to lower area
            nengo.Connection(model.motor.output, model.fingers.input, transform=.2*motor_mapping)

            #finger position (spinal?)
            model.finger_pos = nengo.networks.EnsembleArray(n_neurons=50, n_ensembles=4)
            nengo.Connection(model.finger_pos.output, model.finger_pos.input, synapse=0.1, transform=0.3) #feedback

            #connection between finger area and finger position
            nengo.Connection(model.fingers.am.elem_output, model.finger_pos.input, transform=1.5*np.diag([0.55, .54, .56, .55])) #fix these




        model.bg = spa.BasalGanglia(
            spa.Actions(
                a_attend_item1    =    'dot(goal,DO_TASK) - dot(attend,ITEM1) --> goal=RECOG, attend=ITEM1',
                b_attending_item1 =    'dot(goal,RECOG) + dot(attend,ITEM1) - concepts_evidence - .2 --> goal=RECOG, attend=ITEM1, vis_pair=2*attend*concepts+2*concepts', #, dm_learned_words=vis_pair',
                c_attend_item2    =    'dot(goal,RECOG) + dot(attend,ITEM1) + concepts_evidence - 1.8 --> goal=RECOG, attend=ITEM2, vis_pair=2*attend*concepts+2*concepts, dm_learned_words=vis_pair',
                d_attending_item2 =    'dot(goal,RECOG) + dot(attend,ITEM2) - concepts_evidence - .3 --> goal=RECOG, attend=ITEM2, vis_pair=2*attend*concepts+2*concepts, dm_learned_words=vis_pair',
                e_judge_familiarity =  'dot(goal,RECOG) + dot(attend,ITEM2) + concepts_evidence - 2.1 --> goal=FAMILIARITY, attend=ITEM2, vis_pair=2*attend*concepts+2*concepts, dm_learned_words=vis_pair, do_fam=GO',

                fa_judge_familiarityA = 'dot(goal,FAMILIARITY) - .0 --> goal=FAMILIARITY, dm_learned_words=vis_pair, do_fam=GO',

                g_respond_unfamiliar = 'dot(goal,FAMILIARITY+RESPOND) - familiarity - .9 --> goal=RESPOND, dm_learned_words=vis_pair, do_fam=GO, motor_input=1.5*target_hand+MIDDLE',
                h_respond_familiar =   'dot(goal,FAMILIARITY+RESPOND) + familiarity - .9 --> goal=RESPOND, dm_learned_words=vis_pair, do_fam=GO, motor_input=1.5*target_hand+INDEX,vis_pair=dm_learned_words',




                #fam 'dot(goal,RECOG2)+dot(attend,ITEM2)+familiarity-1.3 --> goal=RECOLLECTION,dm_pairs = 2*vis_pair, representation=3*dm_pairs',# vis_pair=ITEM2*concepts',
                #fam 'dot(goal,RECOLLECTION) - .5 --> goal=RECOLLECTION, representation=2*dm_pairs',

                #fam 'dot(goal,RECOLLECTION) + 2*rep_filled - 1.3 --> goal=COMPARE_ITEM1, attend=ITEM1, comparison_A = 2*vis_pair,comparison_B = 2*representation*~attend',
                #fam 'dot(goal,COMPARE_ITEM1) + rep_filled + comparison -1 --> goal=COMPARE_ITEM2, attend=ITEM2, comparison_A = 2*vis_pair',#comparison_B = 2*representation*~attend',
                #fam 'dot(goal,COMPARE_ITEM1) + rep_filled + (1-comparison) -1 --> goal=RESPOND,motor_input=1.0*target_hand+MIDDLE',#comparison_A = 2*vis_pair,comparison_B = 2*representation*~attend',
                #fam 'dot(goal,COMPARE_ITEM2) + rep_filled + comparison - 1 --> goal=RESPOND,motor_input=1.0*target_hand+INDEX',#comparison_A = 2*vis_pair,comparison_B = 2*representation*~attend',
                #fam 'dot(goal,COMPARE_ITEM2) + rep_filled + (1-comparison) -1 --> goal=RESPOND,motor_input=1.0*target_hand+MIDDLE',#comparison_A = 2*vis_pair,comparison_B = 2*representation*~attend',

                #fam 'dot(goal,RESPOND) + comparison - 1 --> goal=RESPOND, motor_input=1.0*target_hand+INDEX', #comparison_A = 2*vis_pair,comparison_B = 2*representation*~attend',
                #fam 'dot(goal,RESPOND) + (1-comparison) - 1 --> goal=RESPOND, motor_input=1.0*target_hand+MIDDLE', #comparison_A = 2*vis_pair,comparison_B = 2*representation*~attend',

                # 'dot(goal,RECOLLECTION) + (1 - dot(representation,vis_pair)) - 1.3 --> goal=RESPOND, motor_input=1.0*target_hand+MIDDLE',
                x_response_done = 'dot(goal,RESPOND) + dot(motor,MIDDLE+INDEX) - .5 --> goal=END',
                y_end = 'dot(goal,END)-1 --> goal=END',
                z_threshold = '.1 -->'

                #possible to match complete buffer, ie is representation filled?
                # motor_input=1.5*target_hand+MIDDLE,

            ))

        #'dot(attention, W1) - evidence - 0.8 --> motor=NO, attention=W1',
        #'dot(attention, W1) + evidence - 0.8 --> attention=W2, reset=EVIDENCE',
        #'dot(attention, W1) --> attention=W1',  # if we don't set attention it goes back to 0
        #'dot(attention, W2) - evidence - 0.8 --> motor=NO, attention=W2',
        #'dot(attention, W2) + evidence - 0.8 --> motor=YES, attention=W2',
        #'dot(attention, W2) --> attention=W2',  # option might be feedback on attention, then no rule 3/6 but default rule



        model.thalamus = spa.Thalamus(model.bg)

        model.cortical = spa.Cortical( # cortical connection: shorthand for doing everything with states and connections
            spa.Actions(
              #  'motor_input = .04*target_hand',
                #'dm_learned_words = .8*concepts', #.5
                #'dm_pairs = 2*stimulus'
                #'vis_pair = 2*attend*concepts+concepts',
                #fam 'comparison_A = 2*vis_pair',
                #fam 'comparison_B = 2*representation*~attend',

            ))


        #probes
        #model.pr_goal = nengo.Probe(model.goal.output,synapse=.01)
        model.pr_motor_pos = nengo.Probe(model.finger_pos.output,synapse=.01) #raw vector (dimensions x time)
        #model.pr_motor = nengo.Probe(model.fingers.output,synapse=.01)
        #model.pr_motor1 = nengo.Probe(model.motor.output, synapse=.01)
        #model.pr_target = nengo.Probe(model.target_hand.output, synapse=.01)
        #model.pr_attend = nengo.Probe(model.attend.output, synapse=.01)

        #input
        model.input = spa.Input(goal=lambda t: 'DO_TASK' if t < 0.02 else '0',
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
def prepare_sim(seed=None):

    print '---- BUILDING SIMULATOR ----'

    global sim

    print('\t' + str(sum(ens.n_neurons for ens in model.all_ensembles)) + ' neurons')
    print('\t' + str(len(vocab_concepts.keys)) + ' concepts')

    start = time.time()

    if ocl:
        sim = nengo_ocl.Simulator(model,context=ctx)
    else:
        sim = nengo.Simulator(model)
    print('\n ---- DONE in ' + str(round(time.time() - start,2)) + ' seconds ----\n')



total_sim_time = 0

#called by all functions to do a single trial
def do_trial(trial_info, hand):

    global total_sim_time
    global results
    global cur_item1
    global cur_item2
    global cur_hand

    cur_item1 = trial_info[3]
    cur_item2 = trial_info[4]
    cur_hand = hand

    if verbose:
        print('\n\n---- Trial: ' + trial_info[0] + ', Fan ' + str(trial_info[1])
          + ', ' + trial_info[2] + ' - ' + ' '.join(trial_info[3:]) + ' - ' + hand + ' ----\n')

    #run sim at least 100 ms
    sim.run(.1,progress_bar=verbose) #make this shorter than fastest RT

    if verbose:
        print('Stepped sim started...')

    stepsize = 5 #ms
    resp = -1
    while sim.time < 1:

        # run stepsize ms, update time
        sim.run_steps(stepsize, progress_bar=False)

        #target_h = sim.data[model.pr_target][sim.n_steps-1]
        #print np.dot(target_h, vocab_motor['RIGHT'].v)

        #calc finger position
        last_motor_pos = sim.data[model.pr_motor_pos][int(sim.n_steps)-1 ]
        position_finger = np.max(last_motor_pos)

        if position_finger >.68: #.68 represents key press
            if verbose:
                print('... and done!\n')
            break


    # determine response
    step = int(sim.n_steps)
    similarities = [np.dot(sim.data[model.pr_motor][step - 1], vocab_fingers['L1'].v),
                    np.dot(sim.data[model.pr_motor][step - 1], vocab_fingers['L2'].v),
                    np.dot(sim.data[model.pr_motor][step - 1], vocab_fingers['R1'].v),
                    np.dot(sim.data[model.pr_motor][step - 1], vocab_fingers['R2'].v)]
    resp = np.argmax(similarities)

    if verbose:
        if resp == 0:
            print 'Left Index'
        elif resp == 1:
            print 'Left Middle'
        elif resp == 2:
            print 'Right Index'
        elif resp == 3:
            print 'Right Middle'
        if resp == -1:
            print 'No response'


    #resp 0 = left index, 1 = left middle, 2 = right index, 3 = right middle
    #change coding later
    acc = 0 #default 0
    if trial_info[0] == 'Target':
        if (resp == 0 and hand == 'LEFT')  or (resp == 2 and hand == 'RIGHT'):
            acc = 1
    else: #re-paired foil or new foil
        if (resp == 1 and hand == 'LEFT') or (resp == 3 and hand == 'RIGHT'):
            acc = 1

    if verbose:
        print('RT = ' + str(sim.time) + ', acc = ' + str(acc))
    total_sim_time += sim.time
    results.append(trial_info + (hand, np.round(sim.time,3), acc, resp))


def do_1_trial(trial_info=('Target', 1, 'Short', 'METAL', 'SPARK'),hand='RIGHT',subj=0):

    global total_sim_time
    global results
    total_sim_time = 0
    results = []

    start = time.time()

    initialize_model(subj=subj)
    create_model()
    prepare_sim()

    do_trial(trial_info,hand)

    print('\nTotal time: ' + str(round(time.time() - start,2)) + ' seconds for ' + str(round(total_sim_time,2)) + ' seconds simulation.\n')
    sim.close()

    print(results)
    print('\n')


def do_4_trials(subj=0):

    global total_sim_time
    global results
    total_sim_time = 0
    results = []

    start = time.time()

    initialize_model(subj=subj)
    create_model()
    prepare_sim()

    stims_in = []
    for i in [0,33, 32,1]:
        stims_in.append(stims[i])

    hands_in = ['RIGHT','LEFT','RIGHT','LEFT']

    for i in range(4):
        sim.reset() #reset simulator
        do_trial(stims_in[i], hands_in[i])

    print(
    '\nTotal time: ' + str(round(time.time() - start, 2)) + ' seconds for ' + str(round(total_sim_time,2)) + ' seconds simulation.\n')

    sim.close()

    # save behavioral data
    save_results()



def do_1_block(block_hand='RIGHT', subj=0):

    global total_sim_time
    global results
    total_sim_time = 0
    results = []

    start = time.time()

    initialize_model(subj=subj)
    create_model()
    prepare_sim()

    stims_in = stims_target_rpfoils
    nr_trp = len(stims_target_rpfoils)
    nr_nf = nr_trp / 4

    #add new foils
    stims_in = stims_in + random.sample(stims_new_foils, nr_nf)

    #shuffle
    random.shuffle(stims_in)

    for i in stims_in:
        sim.reset()
        do_trial(i, block_hand)


    print(
    '\nTotal time: ' + str(round(time.time() - start, 2)) + ' seconds for ' + str(round(total_sim_time,2)) + ' seconds simulation.\n')

    sim.close()

    # save behavioral data
    save_results('output' + '_' + cur_hand)


def do_experiment(subj=1):

    print('===== RUNNING FULL EXPERIMENT =====')

    #mix of MEG and EEG experiment
    #14 blocks (7 left, 7 right)
    #64 targets/rp-foils per block + 16 new foils (EEG)
    #total number new foils = 14*16=224. We only have 208, but we can repeat some.
    global verbose
    verbose = False
    global total_sim_time
    global results
    total_sim_time = 0
    results = []

    start = time.time()

    initialize_model(subj=subj)
    create_model()
    prepare_sim()

    #split nf in long and short
    nf_short = list()
    nf_long = list()
    for stim in stims_new_foils:
        if stim[2] == 'Short':
            nf_short.append(stim)
        else:
            nf_long.append(stim)

    #add random selection of 16
    nf_short = nf_short + random.sample(nf_short, 8)
    nf_long = nf_long + random.sample(nf_long, 8)

    #shuffle
    random.shuffle(nf_short)
    random.shuffle(nf_long)

    #for each block
    trial = 0
    for bl in range(14):

        # get all targets/rpfoils for each block
        stims_in = stims_target_rpfoils

        #add unique new foils
        stims_in = stims_in + nf_short[:8] + nf_long[:8]
        del nf_short[:8]
        del nf_long[:8]

        #shuffle
        random.shuffle(stims_in)

        #determine hand
        if (bl+subj) % 2 == 0:
            block_hand = 'RIGHT'
        else:
            block_hand = 'LEFT'

        for i in stims_in:
            trial += 1
            print('Trial ' + str(trial) + '/' + str(80*14))
            sim.reset()
            do_trial(i, block_hand)

    print(
    '\nTotal time: ' + str(round(time.time() - start, 2)) + ' seconds for ' + str(round(total_sim_time,2)) + ' seconds simulation.\n')

    sim.close()

    # save behavioral data
    save_results('output' + '_model_subj_' + str(subj))




#choice of trial, etc
if not nengo_gui_on:

    #do_1_trial(trial_info=('Target', 1, 'Short', 'METAL', 'SPARK'), hand='RIGHT')
    #do_1_trial(trial_info=('NewFoil', 1, 'Short', 'CARGO', 'HOOD'),hand='LEFT')
    #do_1_trial(trial_info=('RPFoil', 1,	'Short', 'SODA', 'BRAIN'), hand='RIGHT')

    do_4_trials()

    #do_1_block('RIGHT',subj=0)
    #do_1_block('LEFT')
    #do_experiment(1)


else:
    #nengo gui on

    #New Foils
    #cur_item1 = 'CARGO'
    #cur_item2 = 'HOOD'

    #Targets
    cur_item1 = 'METAL'
    cur_item2 = 'SPARK'

    #Re-paired foisl
    # cur_item1 = 'SODA'
    # cur_item2 = 'BRAIN'


    cur_hand = 'LEFT'

    initialize_model(subj=0)
    create_model()



