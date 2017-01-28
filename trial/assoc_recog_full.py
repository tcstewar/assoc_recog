#nengo
import nengo
import nengo.spa as spa

import pytry

import numpy as np
import os, sys, random, inspect
import itertools
import warnings

verbose = True
fixation_time = 200 #ms

cur_path = '..'

#### HELPER FUNCTIONS ####


#display stimuli in gui, works for 28x90 (two words) and 14x90 (one word)
#t = time, x = vector of pixel values
def display_func(t, x):
    import png
    import PIL.Image
    import base64
    import cStringIO

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
#short=True does the same, except for any random subject, odd subjects get short words, even subjects long words
def load_stims(subj=0,short=True):

    #subj=0 is old, new version makes subj 0 subj 1 + short, but with a fixed stim set
    sub0 = False
    if subj==0:
        sub0 = True
        subj = 1
        short = True

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
    stimsNFlong = np.genfromtxt(cur_path + '/stims/S' + str(subj) + 'NFLong.txt', skip_header=True,
                                dtype=[('probe', np.str_, 8), ('fan', 'int'), ('wordlen', np.str_, 8),
                                       ('item1', np.str_, 8), ('item2', np.str_, 8)], usecols=[1,2,3,4,5])

    #if short, use a small set of new foils
    if short:
        if subj % 2 == 0: #even -> long
            stimsNF = random.sample(stimsNFlong,8)
        else: #odd -> short
            if sub0:  # not random when sub0 == True
                stimsNF = stimsNFshort[0:8]
            else:
                stimsNF = random.sample(stimsNFshort,8)
    else:
        stimsNF = np.hstack((stimsNFshort,stimsNFlong))

    #combine
    stims = np.hstack((stims, stimsNF))
    stims = stims.tolist()

    #if short, only keep shorts for odd subjects or longs for even subjects
    new_stims = []
    if short:
        for i in stims:
            if subj % 2 == 0 and i[2] == 'Long':
                new_stims.append(i)
            elif subj % 2 == 1 and i[2] == 'Short':
                new_stims.append(i)
        stims = new_stims

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
            target_words.append(i[3])
            target_words.append(i[4])
        else:
            newfoil_pairs.append((i[3],i[4]))


        # make separate lists for targets/rp foils and new foils (for presenting experiment)
        if i[0] != 'NewFoil':
            stims_target_rpfoils.append(i)
        else:
            stims_new_foils.append(i)

    # remove duplicates
    items = np.unique(items).tolist()
    #items.append('FIXATION')
    target_words = np.unique(target_words).tolist()



# load images for vision
def load_images():
    import png

    global X_train, y_train, y_train_words, fixation_image

    indir = cur_path + '/images/'
    files = os.listdir(indir)
    files2 = []

    #select only images for current item set
    for fn in files:
        if fn[-4:] == '.png' and ((fn[:-4] in items)):
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


    #add fixation separately (only for presenting, no mapping to concepts)
    r = png.Reader(cur_path + '/images/FIXATION.png')
    r = r.asDirect()
    image_2d = np.vstack(itertools.imap(np.uint8, r[2]))
    image_2d /= 255
    fixation_image = np.empty(shape=(1,90*14),dtype='float32')
    fixation_image[0] = image_2d.reshape(1, 90 * 14)

def load_image_words():
    global y_train_words
    indir = cur_path + '/images/'
    files = os.listdir(indir)
    files2 = []

    #select only images for current item set
    for fn in files:
        if fn[-4:] == '.png' and ((fn[:-4] in items)):
             files2.append(fn)

    y_train_words = [] #words represented in X_train
    for i,fn in enumerate(files2):
        y_train_words.append(fn[:-4]) #add word

#returns pixels of image representing item (ie METAL)
def get_image(item):
    if item != 'FIXATION':
        return X_train[y_train_words.index(item)]
    else:
        return fixation_image[0]



#### MODEL FUNCTIONS #####


# performs all steps in model ini
def initialize_model(p):
    #warn when loading full stim set with low dimensions:
    if not(p.short) and not(high_dims):
        warn = warnings.warn('Initializing model with full stimulus set, but using low dimensions for vocabs.')

    load_stims(p.subj,short=p.short)
    if p.do_vision:
        load_images()
    else:
        load_image_words()
    initialize_vocabs(p)


#initialize vocabs
def initialize_vocabs(p):

    print('---- INITIALIZING VOCABS ----')

    global vocab_vision #low level visual vocab
    global vocab_concepts #vocab with all concepts and pairs
    global vocab_learned_words #vocab with learned words
    global vocab_all_words #vocab with all words
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
    vocab_vision = nengo.spa.Vocabulary(p.Dmid,max_similarity=.25)
    for name in y_train_words:
        vocab_vision.parse(name)
    train_targets = vocab_vision.vectors


    #word concepts - has all concepts, including new foils, and pairs
    vocab_concepts = spa.Vocabulary(p.D, max_similarity=0.1)
    for i in y_train_words:
        vocab_concepts.parse(i)
    vocab_concepts.parse('ITEM1')
    vocab_concepts.parse('ITEM2')
    vocab_concepts.parse('NONE')

    list_of_pairs = []
    for item1, item2 in target_pairs:
        vocab_concepts.add('%s_%s' % (item1, item2), vocab_concepts.parse(
            '%s*ITEM1 + %s*ITEM2' % (item1, item2)))  # add pairs to concepts to use same vectors
        list_of_pairs.append('%s_%s' % (item1, item2))  # keep list of pairs notation

    # add all presented pairs to concepts for display
    for item1, item2 in newfoil_pairs:
        vocab_concepts.add('%s_%s' % (item1, item2), vocab_concepts.parse(
            '%s*ITEM1 + %s*ITEM2' % (item1, item2)))  # add pairs to concepts to use same vectors
    for item1, item2 in rpfoil_pairs:
        vocab_concepts.add('%s_%s' % (item1, item2), vocab_concepts.parse(
            '%s*ITEM1 + %s*ITEM2' % (item1, item2)))  # add pairs to concepts to use same vectors


    #vision-concept mapping between vectors
    vision_mapping = np.zeros((p.D, p.Dmid))
    for word in y_train_words:
        vision_mapping += np.outer(vocab_vision.parse(word).v, vocab_concepts.parse(word).v).T

    #vocab with learned words
    vocab_learned_words = vocab_concepts.create_subset(target_words + ['NONE'])

    #vocab with all words
    vocab_all_words = vocab_concepts.create_subset(y_train_words + ['ITEM1', 'ITEM2'])

    #vocab with learned pairs
    vocab_learned_pairs = vocab_concepts.create_subset(list_of_pairs) #get only pairs

    #print vocab_learned_words.keys
    #print y_train_words
    #print target_words

    #motor vocabs, just for sim calcs
    vocab_motor = spa.Vocabulary(p.Dmid) #different dimension to be sure, upper motor hierarchy
    vocab_motor.parse('LEFT+RIGHT+INDEX+MIDDLE')

    vocab_fingers = spa.Vocabulary(p.Dlow) #direct finger activation
    vocab_fingers.parse('L1+L2+R1+R2')

    #map higher and lower motor
    motor_mapping = np.zeros((p.Dlow, p.Dmid))
    motor_mapping += np.outer(vocab_motor.parse('LEFT+INDEX').v, vocab_fingers.parse('L1').v).T
    motor_mapping += np.outer(vocab_motor.parse('LEFT+MIDDLE').v, vocab_fingers.parse('L2').v).T
    motor_mapping += np.outer(vocab_motor.parse('RIGHT+INDEX').v, vocab_fingers.parse('R1').v).T
    motor_mapping += np.outer(vocab_motor.parse('RIGHT+MIDDLE').v, vocab_fingers.parse('R2').v).T

    #goal vocab
    vocab_goal = spa.Vocabulary(p.Dlow)
    vocab_goal.parse('DO_TASK')
    vocab_goal.parse('RECOG')
    vocab_goal.parse('RECOG2')
    vocab_goal.parse('FAMILIARITY')
    vocab_goal.parse('RESPOND')
    vocab_goal.parse('END')

    #attend vocab
    vocab_attend = vocab_concepts.create_subset(['ITEM1', 'ITEM2'])

    #reset vocab
    vocab_reset = spa.Vocabulary(p.Dlow)
    vocab_reset.parse('CLEAR+GO')



# word presented in current trial
global cur_item1
global cur_item2
global cur_hand
cur_item1 = 'METAL' #just for ini
cur_item2 = 'SPARK'
cur_hand = 'LEFT'

# returns images of current words for display # fixation for 51 ms.
def present_pair(t):
    if t < (fixation_time/1000.0)+.002:
        return np.hstack((np.ones(7*90),get_image('FIXATION'),np.ones(7*90)))
    else:
        im1 = get_image(cur_item1)
        im2 = get_image(cur_item2)
        return np.hstack((im1, im2))



def present_item2(t, output_attend):

    #no-attention scale factor
    no_attend = .1

    #first fixation before start trial
    if t < (fixation_time/1000.0) + .002:
        # ret_ima = np.zeros(1260)
        ret_ima = no_attend * get_image('FIXATION')
    else: #then either word or zeros (mix of words?)
        attn = vocab_attend.dot(output_attend) #dot product with current input (ie ITEM 1 or 2)
        i = np.argmax(attn) #index of max

        #ret_ima = np.zeros(1260)

        ret_ima = no_attend * (get_image(cur_item1) + get_image(cur_item2)) / 2

        if attn[i] > 0.3: #if we really attend something
            if i == 0: #first item
                ret_ima = get_image(cur_item1)
            else:
                ret_ima = get_image(cur_item2)

    return (.8 * ret_ima)

def fake_vision(t, output_attend):
    #first fixation before start trial
    if t < (fixation_time/1000.0) + .002:
        v = '0'
    else: #then either word or zeros (mix of words?)
        attn = vocab_attend.dot(output_attend) #dot product with current input (ie ITEM 1 or 2)
        i = np.argmax(attn) #index of max

        v = '0'

        if attn[i] > 0.3: #if we really attend something
            if i == 0: #first item
                v = cur_item1
            else:
                v = cur_item2
    return vocab_all_words.parse(v).v


#get vector representing hand
def get_hand(t):
    #print(cur_hand)
    return vocab_motor.vectors[vocab_motor.keys.index(cur_hand)]



def create_model(p):
    model = spa.SPA()

    if p.direct:
        model.config[nengo.Ensemble].neuron_type = nengo.Direct()
        force_neurons_cfg = nengo.Config(nengo.Ensemble)
        force_neurons_cfg[nengo.Ensemble].neuron_type = nengo.LIF()
    else:
        force_neurons_cfg = model.config

    with model:

        # control
        model.control_net = nengo.Network()
        with model.control_net:
            #assuming the model knows which hand to use (which was blocked)
            model.hand_input = nengo.Node(get_hand)
            model.target_hand = spa.State(p.Dmid, vocab=vocab_motor, feedback=1)
            nengo.Connection(model.hand_input,model.target_hand.input,synapse=None)

            model.attend = spa.State(p.D, vocab=vocab_attend, feedback=.5)  # vocab_attend
            model.goal = spa.State(p.Dlow, vocab=vocab_goal, feedback=.7)  # current goal

        if p.do_vision:
            from nengo_extras.vision import Gabor, Mask


            ### vision ###

            # set up network parameters
            n_vis = X_train.shape[1]  # nr of pixels, dimensions of network
            n_hid = 1000  # nr of gabor encoders/neurons

            # random state to start
            rng = np.random.RandomState(9)
            encoders = Gabor().generate(n_hid, (4, 4), rng=rng)  # gabor encoders, 11x11 apparently, why?
            encoders = Mask((14, 90)).populate(encoders, rng=rng,
                                               flatten=True)  # use them on part of the image


            model.visual_net = nengo.Network()
            with model.visual_net:

                #represent currently attended item
                model.attended_item = nengo.Node(present_item2,size_in=p.D)
                nengo.Connection(model.attend.output, model.attended_item)

                model.vision_gabor = nengo.Ensemble(n_hid, n_vis, eval_points=X_train,
                                                    #    neuron_type=nengo.LIF(),
                                                        neuron_type=nengo.AdaptiveLIF(tau_n=.01, inc_n=.05), #to get a better fit, use more realistic neurons that adapt to input
                                                        intercepts=nengo.dists.Uniform(-0.1, 0.1),
                                                        #intercepts=nengo.dists.Choice([-0.5]), #should we comment this out? not sure what's happening
                                                        #max_rates=nengo.dists.Choice([100]),
                                                        encoders=encoders)
                #recurrent connection (time constant 500 ms)
                # strength = 1 - (100/500) = .8

                zeros = np.zeros_like(X_train)
                nengo.Connection(model.vision_gabor, model.vision_gabor, synapse=0.005, #.1
                                 eval_points=np.vstack([X_train, zeros, np.random.randn(*X_train.shape)]),
                                 transform=.5)

                model.visual_representation = nengo.Ensemble(n_hid, dimensions=p.Dmid)

                model.visconn = nengo.Connection(model.vision_gabor, model.visual_representation, synapse=0.005, #was .005
                                                eval_points=X_train, function=train_targets,
                                                solver=nengo.solvers.LstsqL2(reg=0.01))
                nengo.Connection(model.attended_item, model.vision_gabor, synapse=.02) #.03) #synapse?

                # display attended item, only in gui
                if p.gui:
                    # show what's being looked at
                    model.display_attended = nengo.Node(display_func, size_in=model.attended_item.size_out)  # to show input
                    nengo.Connection(model.attended_item, model.display_attended, synapse=None)
                    #add node to plot total visual activity
                    model.visual_activation = nengo.Node(None,size_in=1)
                    nengo.Connection(model.vision_gabor.neurons, model.visual_activation,transform=np.ones((1,n_hid)), synapse = None)
        else:
            model.visual_net = nengo.Network()
            with model.visual_net:
                model.fake_vision = nengo.Node(fake_vision, size_in=p.D)
            nengo.Connection(model.attend.output, model.fake_vision)



        ### central cognition ###

        ##### Concepts #####
        model.concepts = spa.AssociativeMemory(vocab_all_words, #vocab_concepts,
                                               wta_output=True,
                                               wta_inhibit_scale=1, #was 1
                                               #default_output_key='NONE', #what to say if input doesn't match
                                               threshold=0.3)  # how strong does input need to be for it to recognize
        if p.do_vision:
            nengo.Connection(model.visual_representation, model.concepts.input, transform=.8*vision_mapping) #not too fast to concepts, might have to be increased to have model react faster to first word.
        else:
            nengo.Connection(model.fake_vision, model.concepts.input, transform=.8) #not too fast to concepts, might have to be increased to have model react faster to first word.

        #concepts accumulator
        with force_neurons_cfg:
            model.concepts_evidence = spa.State(1, feedback=1, feedback_synapse=0.005) #the lower the synapse, the faster it accumulates (was .1)
        concepts_evidence_scale = 2.5
        nengo.Connection(model.concepts.am.elem_output, model.concepts_evidence.input,
                         transform=concepts_evidence_scale * np.ones((1, model.concepts.am.elem_output.size_out)),synapse=0.005)

        #concepts switch
        model.do_concepts = spa.AssociativeMemory(vocab_reset, default_output_key='CLEAR', threshold=.2)
        nengo.Connection(model.do_concepts.am.ensembles[-1], model.concepts_evidence.all_ensembles[0].neurons,
                         transform=np.ones((model.concepts_evidence.all_ensembles[0].n_neurons, 1)) * -10,
                         synapse=0.005)

        ###### Visual Representation ######
        model.vis_pair = spa.State(p.D, vocab=vocab_all_words, feedback=1.0, feedback_synapse=.05) #was 2, 1.6 works ok, but everything gets activated.

        model.p_vis_pair = nengo.Probe(model.vis_pair.output, synapse=0.01)

        if p.do_familiarity:
            assert p.do_motor
                        ##### Familiarity #####
            # Assoc Mem with Learned Words
            # - familiarity signal should be continuous over all items, so no wta
            model.dm_learned_words = spa.AssociativeMemory(vocab_learned_words,threshold=.2)
            nengo.Connection(model.dm_learned_words.output,model.dm_learned_words.input,transform=.4,synapse=.02)

            # Familiarity Accumulator

            with force_neurons_cfg:
                model.familiarity = spa.State(1, feedback=.9, feedback_synapse=0.1) #fb syn influences speed of acc
            #familiarity_scale = 0.2 #keep stable for negative fam

            # familiarity accumulator switch
            model.do_fam = spa.AssociativeMemory(vocab_reset, default_output_key='CLEAR', threshold=.2)
            # reset
            nengo.Connection(model.do_fam.am.ensembles[-1], model.familiarity.all_ensembles[0].neurons,
                             transform=np.ones((model.familiarity.all_ensembles[0].n_neurons, 1)) * -10,
                             synapse=0.005)


            #first a sum to represent summed similarity
            model.summed_similarity = nengo.Ensemble(n_neurons=100,dimensions=1)
            nengo.Connection(model.dm_learned_words.am.elem_output, model.summed_similarity,
                    transform=np.ones((1, model.dm_learned_words.am.elem_output.size_out))) #take sum

            #then a connection to accumulate this summed sim
            def familiarity_acc_transform(summed_sim):
                    fam_scale = .5
                    fam_threshold = 0 #actually, kind of bias
                    fam_max = 1
                    return fam_scale*(2*((summed_sim - fam_threshold)/(fam_max - fam_threshold)) - 1)

            nengo.Connection(model.summed_similarity, model.familiarity.input,
                             function=familiarity_acc_transform)


            ##### Recollection & Representation #####

            model.dm_pairs = spa.AssociativeMemory(vocab_learned_pairs,wta_output=True) #input_keys=list_of_pairs
            nengo.Connection(model.dm_pairs.output,model.dm_pairs.input,transform=.5,synapse=.05)

                    #representation
            rep_scale = 0.5
            model.representation = spa.State(p.D,vocab=vocab_all_words,feedback=1.0)
            with force_neurons_cfg:
                model.rep_filled = spa.State(1,feedback=.9,feedback_synapse=.1) #fb syn influences speed of acc
            model.do_rep = spa.AssociativeMemory(vocab_reset, default_output_key='CLEAR', threshold=.2)
            nengo.Connection(model.do_rep.am.ensembles[-1], model.rep_filled.all_ensembles[0].neurons,
                             transform=np.ones((model.rep_filled.all_ensembles[0].n_neurons, 1)) * -10,
                             synapse=0.005)

            nengo.Connection(model.representation.output, model.rep_filled.input,
                            transform=rep_scale*np.reshape(sum(vocab_learned_pairs.vectors),((1,p.D))))


            ###### Comparison #####

            model.comparison = spa.Compare(p.D, vocab=vocab_all_words,neurons_per_multiply=500, input_magnitude=.3)

            #turns out comparison is not an accumulator - we also need one of those.
            with force_neurons_cfg:
                model.comparison_accumulator = spa.State(1, feedback=.9, feedback_synapse=0.05) #fb syn influences speed of acc
            model.do_compare = spa.AssociativeMemory(vocab_reset, default_output_key='CLEAR', threshold=.2)

                    #reset
            nengo.Connection(model.do_compare.am.ensembles[-1], model.comparison_accumulator.all_ensembles[0].neurons,
                             transform=np.ones((model.comparison_accumulator.all_ensembles[0].n_neurons, 1)) * -10,
                             synapse=0.005)

            #error because we apply a function to a 'passthrough' node, inbetween ensemble as a solution:
            model.comparison_result = nengo.Ensemble(n_neurons=100,dimensions=1)
            nengo.Connection(model.comparison.output, model.comparison_result)

            def comparison_acc_transform(comparison):
                    comparison_scale = .6
                    comparison_threshold = 0 #actually, kind of bias
                    comparison_max = .6
                    return comparison_scale*(2*((comparison - comparison_threshold)/(comparison_max - comparison_threshold)) - 1)


            nengo.Connection(model.comparison_result, model.comparison_accumulator.input,
                             function=comparison_acc_transform)


        if p.do_motor:

            #motor
            model.motor_net = nengo.Network()
            with model.motor_net:

                #input multiplier
                model.motor_input = spa.State(p.Dmid,vocab=vocab_motor)

                #higher motor area (SMA?)
                model.motor = spa.State(p.Dmid, vocab=vocab_motor,feedback=.7)

                #connect input multiplier with higher motor area
                nengo.Connection(model.motor_input.output,model.motor.input,synapse=.1,transform=2)

                #finger area
                model.fingers = spa.AssociativeMemory(vocab_fingers, input_keys=['L1', 'L2', 'R1', 'R2'], wta_output=True)
                nengo.Connection(model.fingers.output, model.fingers.input, synapse=0.1, transform=0.3) #feedback

                #conncetion between higher order area (hand, finger), to lower area
                nengo.Connection(model.motor.output, model.fingers.input, transform=.25*motor_mapping) #was .2

                #finger position (spinal?)
                model.finger_pos = nengo.networks.EnsembleArray(n_neurons=50, n_ensembles=4)
                nengo.Connection(model.finger_pos.output, model.finger_pos.input, synapse=0.1, transform=0.8) #feedback

                #connection between finger area and finger position
                nengo.Connection(model.fingers.am.elem_output, model.finger_pos.input, transform=1.0*np.diag([0.55, .54, .56, .55])) #fix these


        actions = dict(
            #wait & start
            a_aa_wait =            'dot(goal,WAIT) - .9 --> goal=0',
            a_attend_item1    =    'dot(goal,DO_TASK) - .0 --> goal=RECOG, attend=ITEM1, do_concepts=GO',

            #attend words
            b_attending_item1 =    'dot(goal,RECOG) + dot(attend,ITEM1) - concepts_evidence - .3 --> goal=RECOG, attend=ITEM1, do_concepts=GO', # vis_pair=2.5*(ITEM1*concepts)',
            c_attend_item2    =    'dot(goal,RECOG) + dot(attend,ITEM1) + concepts_evidence - 1.6 --> goal=RECOG2, attend=ITEM2, vis_pair=3*(ITEM1*concepts)',

            d_attending_item2 =    'dot(goal,RECOG2+RECOG) + dot(attend,ITEM2) - concepts_evidence - .4 --> goal=RECOG2, attend=ITEM2, do_concepts=GO',
            e_start_familiarity =  'dot(goal,RECOG2) + dot(attend,ITEM2) + concepts_evidence - 1.8 --> goal=FAMILIARITY, vis_pair=1.9*(ITEM2*concepts)',
            )
        if p.do_familiarity:
            actions['d_attending_item2'] += ', dm_learned_words=1.0*(~ITEM1*vis_pair)'
            actions['e_start_familiarity'] += ', do_fam=GO, dm_learned_words=2.0*(~ITEM1*vis_pair+~ITEM2*vis_pair)'
            actions.update(dict(
                #judge familiarity
                f_accumulate_familiarity =  '1.1*dot(goal,FAMILIARITY) - 0.2 --> goal=FAMILIARITY-RECOG2, do_fam=GO, dm_learned_words=.8*(~ITEM1*vis_pair+~ITEM2*vis_pair)',

                g_respond_unfamiliar = 'dot(goal,FAMILIARITY) - familiarity - .5*dot(fingers,L1+L2+R1+R2) - .6 --> goal=RESPOND_MISMATCH-FAMILIARITY, do_fam=GO, motor_input=1.6*(target_hand+MIDDLE)',
                #g2_respond_familiar =   'dot(goal,FAMILIARITY) + familiarity - .5*dot(fingers,L1+L2+R1+R2) - .6 --> goal=RESPOND, do_fam=GO, motor_input=1.6*(target_hand+INDEX)',

                #recollection & representation
                h_recollection =        'dot(goal,FAMILIARITY) + familiarity - .5*dot(fingers,L1+L2+R1+R2) - .6 --> goal=RECOLLECTION-FAMILIARITY, dm_pairs = vis_pair',
                i_representation =      'dot(goal,RECOLLECTION) - rep_filled - .1 --> goal=RECOLLECTION, dm_pairs = vis_pair, representation=3*dm_pairs, do_rep=GO',

                #comparison & respond
                j_10_compare_word1 =    'dot(goal,RECOLLECTION+1.4*COMPARE_ITEM1) + rep_filled - .9 --> goal=COMPARE_ITEM1-RECOLLECTION, do_rep=GO, do_compare=GO, comparison_A = ~ITEM1*vis_pair, comparison_B = ~ITEM1*representation',
                k_11_match_word1 =      'dot(goal,COMPARE_ITEM1) + comparison_accumulator - .7 --> goal=COMPARE_ITEM2-COMPARE_ITEM1, do_rep=GO, comparison_A = ~ITEM1*vis_pair, comparison_B = ~ITEM1*representation',
                l_12_mismatch_word1 =   'dot(goal,COMPARE_ITEM1) + .4 * dot(goal,RESPOND_MISMATCH) - comparison_accumulator - .7 --> goal=RESPOND_MISMATCH-COMPARE_ITEM1, do_rep=GO, motor_input=1.6*(target_hand+MIDDLE), do_compare=GO, comparison_A = ~ITEM1*vis_pair, comparison_B = ~ITEM1*representation',

                compare_word2 =         'dot(goal,COMPARE_ITEM2) - .5 --> goal=COMPARE_ITEM2, do_compare=GO, comparison_A = ~ITEM2*vis_pair, comparison_B = ~ITEM2*representation',
                m_match_word2 =         'dot(goal,COMPARE_ITEM2) + comparison_accumulator - .7 --> goal=RESPOND_MATCH-COMPARE_ITEM2, motor_input=1.6*(target_hand+INDEX), do_compare=GO, comparison_A = ~ITEM2*vis_pair, comparison_B = ~ITEM2*representation',
                n_mismatch_word2 =      'dot(goal,COMPARE_ITEM2) - comparison_accumulator - dot(fingers,L1+L2+R1+R2)- .7 --> goal=RESPOND_MISMATCH-COMPARE_ITEM2, motor_input=1.6*(target_hand+MIDDLE),do_compare=GO, comparison_A = ~ITEM2*vis_pair, comparison_B = ~ITEM2*representation',

                #respond
                o_respond_match =       'dot(goal,RESPOND_MATCH) - .1 --> goal=RESPOND_MATCH, motor_input=1.6*(target_hand+INDEX)',
                p_respond_mismatch =    'dot(goal,RESPOND_MISMATCH) - .1 --> goal=RESPOND_MISMATCH, motor_input=1.6*(target_hand+MIDDLE)',

                #finish
                x_response_done =       'dot(goal,RESPOND_MATCH) + dot(goal,RESPOND_MISMATCH) + 2*dot(fingers,L1+L2+R1+R2) - .7 --> goal=2*END',
                y_end =                 'dot(goal,END)-.1 --> goal=END-RESPOND_MATCH-RESPOND_MISMATCH',
                z_threshold =           '.05 --> goal=0'

                #possible to match complete buffer, ie is representation filled?
                # motor_input=1.5*target_hand+MIDDLE,
                ))

        with force_neurons_cfg:
            model.bg = spa.BasalGanglia(spa.Actions(**actions))


        with force_neurons_cfg:
            model.thalamus = spa.Thalamus(model.bg)

        model.p_bg_input = nengo.Probe(model.bg.input)
        model.p_thal_output = nengo.Probe(model.thalamus.actions.output,
                                          synapse=0.01)


        #probes
        if p.do_motor:
            model.pr_motor_pos = nengo.Probe(model.finger_pos.output,synapse=.01) #raw vector (dimensions x time)
            model.pr_motor = nengo.Probe(model.fingers.output,synapse=.01)
        #model.pr_motor1 = nengo.Probe(model.motor.output, synapse=.01)

        #if not p.gui:
        #    model.pr_vision_gabor = nengo.Probe(model.vision_gabor.neurons,synapse=.005) #do we need synapse, or should we do something with the spikes
        #    model.pr_familiarity = nengo.Probe(model.dm_learned_words.am.elem_output,synapse=.01) #element output, don't include default
        #    model.pr_concepts = nengo.Probe(model.concepts.am.elem_output, synapse=.01)  # element output, don't include default

        #multiply spikes with the connection weights


        #input
        model.input = spa.Input(goal=goal_func)

        if p.gui:
            vocab_actions = spa.Vocabulary(model.bg.output.size_out)
            for i, action in enumerate(model.bg.actions.actions):
                vocab_actions.add(action.name.upper(), np.eye(model.bg.output.size_out)[i])
            model.actions = spa.State(model.bg.output.size_out,subdimensions=model.bg.output.size_out,
                                  vocab=vocab_actions)
            nengo.Connection(model.thalamus.output, model.actions.input)

            for net in model.networks:
                if net.label is not None and net.label.startswith('channel'):
                    net.label = ''
    return model


def goal_func(t):
    if t < (fixation_time/1000.0) + .002:
        return 'WAIT'
    elif t < (fixation_time/1000.0) + .022: #perhaps get from distri
        return 'DO_TASK'
    else:
        return '0'  # first 50 ms fixation


class AssocRecogTrial(pytry.NengoTrial):
    def params(self):
        self.param('direct mode', direct=False)
        self.param('include detailed vision', do_vision=True)
        self.param('include detailed motor', do_motor=True)
        self.param('perform familiarity', do_familiarity=True)
        self.param('subject', subj=0)
        self.param('short words', short=True)
        self.param('dimensionality', D=256)
        self.param('dimensionality of vision and motor maps', Dmid=48)
        self.param('dimensionality of low-level reps', Dlow=48)

    def model(self, p):
        cur_item1 = 'CARGO'
        cur_item2 = 'HOOD'

        #New Foils2
        #cur_item1 = 'EXIT'
        #cur_item2 = 'BARN'

        #Targets
        #cur_item1 = 'METAL'
        #cur_item2 = 'SPARK'

        #Re-paired foils 1
        #cur_item1 = 'SODA' #matches HERB-BRAIN, so first one is mismatch, might be coincidence though
        #cur_item2 = 'BRAIN'

        #Re-paired foils 2
        #cur_item1 = 'METAL' #matches METAL-SPARK, so second one is mismatch (again, this is coincidence)
        #cur_item2 = 'MOTOR'

        #Re-paired foils 3
        #cur_item1 = 'JELLY' #matches METAL-SPARK, so second one is mismatch (again, this is coincidence)
        #cur_item2 = 'SPARK'

        #Re-paired foils 4
        #cur_item1 = 'DEBT' #matches METAL-SPARK, so second one is mismatch (again, this is coincidence)
        #cur_item2 = 'SPEAR'

        cur_hand = 'RIGHT'

        initialize_model(p)

        model = create_model(p)

        for name, p in inspect.getmembers(model, lambda x: isinstance(x, nengo.Probe)):
            setattr(self, name, p)

        return model

    def evaluate(self, p, sim, plt):
        T = 0.5
        if p.do_familiarity:
            T = 3.0
        sim.run(T)

        if plt is not None:
            N = 3
            plt.subplot(N,1,1)
            plt.plot(sim.trange(), sim.data[self.p_bg_input])
            plt.ylabel('bg input')
            plt.ylim(0, 1.6)

            plt.subplot(N,1,2)
            plt.plot(sim.trange(), sim.data[self.p_thal_output])
            plt.ylabel('thal output')
            plt.ylim(0, 1.0)

            plt.subplot(N, 1, 3)
            p1 = vocab_all_words.parse('%s*ITEM1' % cur_item1).v
            p2 = vocab_all_words.parse('%s*ITEM2' % cur_item2).v
            data = sim.data[self.p_vis_pair]
            plt.plot(sim.trange(), np.dot(data, p1), label=cur_item1)
            plt.plot(sim.trange(), np.dot(data, p2), label=cur_item2)
            plt.legend(loc='best')
            plt.ylabel('vis_pair')
            plt.ylim(-0.2, 1.4)
            plt.text(0, 0, 'do_vision=%s' % p.do_vision)


        return {}








