#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2022.1.2),
    on Wed Jun  8 16:22:57 2022
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

from psychopy import locale_setup
from psychopy import prefs
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

from psychopy.hardware import keyboard

### BEGIN TRANSLATION ###
# intro screen text
INTRO_TXT = (
    "I de här uppgifterna ska du hitta figurer i olika bilder. " 
    "På vissa bilder ska du hitta figurer med samma form flera gånger och " 
    "på vissa bilder ska du hitta flera olika figurer som presenteras en i taget. \n\n" 
    "Figuren som du ska leta efter visas ovanför bilden." 
    " Den visas upp till 90 sekunder när du ska hitta likadana figurer " 
    "och upp till 30 sekunder för varje figur när du ska hitta olika. \n\n" 
    "Figuren som du ska leta efter ska ha exakt samma form men kan presenteras"
    " i olika storlekar och vinklar.\n\n"
    "Varje gång du hittar en figur klickar du på den."
)

# continue button text
CONTINUE_TXT = "Fortsätt"

# first training 'trial' instructions text
TRAIN1_TXT_INST = (
    "Här ser du ett exempel på en av uppgifterna. \n"
    "Titta på figuren ovanför bilden och försök hitta den i bilden. \n"
    "Kom ihåg att figuren ska ha exakt samma form som den som presenteras ovanför "
    "men kan ha en annan storlek och vinkel. \n"
    "Hela figuren kommer att synas men det kan finnas andra figurer inom den."
)

# first training 'trial' feedback for correct response
TRAIN1_TXT_CORR = (
    "Det var rätt!"
)

# first training 'trial' feedback for incorrect response
TRAIN1_TXT_INCORR = (
    "Du hittade inte rätt figur: figuren du ska hitta ska ha "
    "exakt samma form som figuren ovanpå. Se "
    "markeringen för rätt svar."
)

# second training 'trial' instructions text
TRAIN2_TXT_INST = (
    "Här ser du ett annat exempel på uppgift. "
    "Den här gången ska du hitta flera likadana figurer. "
    "Titta på figuren bredvid bilden och försök sedan hitta fler i bilden."
    "Kom ihåg att figurerna ska ha exakt samma form som den som presenteras "
    "bredvid men kan ha andra storlekar och vinklar.\n\n"
    "Varje gång du hittar en figur klickar du på den."
)

# second training 'trial' feedback for correct response
TRAIN2_TXT_CORR = (
    "Bra! Du hittade alla figurer och klickade inte på fel figur.\n\n"
     "Lägg märke till att det rödmarkerade föremålet "
    "liknar figuren, men har inte exakt samma form."
)

# second training 'trial' feedback for incorrect response
TRAIN2_TXT_INCORR = (
    "Du hittade alla figurer men klickade också på fel figur. " 
    "Figuren du ska hitta ska ha exakt samma form " 
    "som figuren bredvid.\n\n"
    "Lägg märke till att det rödmarkerade föremålet "
    "liknar figuren, men har inte exakt samma form."
)

# (after-training and immediately before first 'real' trial)
# start screen instructions text
START_TXT_INST = (
    "Nu börjar uppgiften. \n\n"
    "Försök att svara så snabbt och att klicka så "
    "exakt på figurerna som möjligt. \n\n"
    "Ett blått kryss visas på platsen där du klickat, oavsett om du klickar "
    "rätt eller fel. Du kan inte ångra om du klickar fel. "
    "Om du klickar fel, fortsätt bara med uppgiften."
)

# start test button text
START_TXT_BTN = (
    "Starta"
)

# end screen message
END_TXT = (
    "Bra jobbat! Nu kommer nästa experiment. När du är redo matar in ditt deltagar-ID för nästa test. Vänligen stanna i samma position. Ifall du behöver lämna din position behöver kalibreringen göras om. I så fall vänligen kontakta experimentledaren, tack!"
)

# trial instruction shown before 'repeated' and 'heap' trials
REP_TXT = (
    "Hitta likadana."
)

# trial instruction shown before 'once' trials
ONCE_TXT = (
    "Hitta olika, en i taget."
)

# trial instruction shown just before sequence of
#'popout' trials starts
POPOUT_TXT = (
    "Nu ska du klicka på figurer som visas helt fristående "
    "från de andra figurerna i bilden.\n"
    "Kom ihåg att figuren som du letar efter ska "
    "se exakt likadan ut som den som "
    "presenteras ovanför bilden men kan ha "
    "en annan storlek eller vinkel. "
    "Figuren visas ovanför bilden i 30 sekunder."
)

## READ MESSAGES AUDIO SECTION ##
# **relative** path to directory holding audio files with
# read instructions/messages to participant
READ_AUDIO_DIR_PATH = (
    "stimuli/audio_read_instructions/swedish"
)

# read messages audio file names
# and durations (in seconds)
# intro screen 
READ_AUDIO_INTRO_FNAME = 'read_intro.wav'
READ_AUDIO_INTRO_DUR = 42.67827664399093
# training 1 instructions
READ_AUDIO_TRAIN1_INST_FNAME = 'read_train1_inst.wav'
READ_AUDIO_TRAIN1_INST_DUR = 22.523356009070294
# training 1 correct
READ_AUDIO_TRAIN1_CORR_FNAME = 'read_train1_correct.wav'
READ_AUDIO_TRAIN1_CORR_DUR = 1.5
# training 1 incorrect
READ_AUDIO_TRAIN1_INCORR_FNAME = 'read_train1_incorrect.wav'
READ_AUDIO_TRAIN1_INCORR_DUR = 7.9412244897959186
# training 2 instructions
READ_AUDIO_TRAIN2_INST_FNAME = 'read_train2_inst.wav'
READ_AUDIO_TRAIN2_INST_DUR = 25.26331065759637
# training 2 correct
READ_AUDIO_TRAIN2_CORR_FNAME = 'read_train2_correct.wav'
READ_AUDIO_TRAIN2_CORR_DUR = 5.6656689342403626
# training 2 incorrect
READ_AUDIO_TRAIN2_INCORR_FNAME = 'read_train2_incorrect.wav'
READ_AUDIO_TRAIN2_INCORR_DUR = 9.357641723356009
# (after-training and immediately before first 'real' trial)
# start screen
READ_AUDIO_START_FNAME = 'read_start.wav'
READ_AUDIO_START_DUR = 20.06204081632653

# before sequence of
#'popout' trials starts
READ_AUDIO_INTRO_FNAME = 'read_train_pre_popout.wav'
READ_AUDIO_INTRO_DUR = 20.135

## END READ MESSAGES AUDIO SECTION ## 

### END TRANSLATION ###
# import psychopy module for generating
# monitor instances.
from psychopy import monitors

# UPDATE HERE create monitor instance for your
# laboratory's monitor.
experiment_mon = monitors.Monitor(
    name="experiment_monitor",
    width=30, # UPDATEME monitor width in centimers
    distance=60, #UPDATEME distance from monitor in centimeters
    notes="Our lab's experiment monitor.")

# UPDATE HERE let PsychoPy know the pixel width/height of
# your laboratory's monitor.
experiment_mon.setSizePix([1280, 1024]) #UPDATEME

# info for pixel to degree conversion
px_per_cm = experiment_mon.getSizePix()[0]/experiment_mon.getWidth()
cm_per_deg = tan(pi / 180) * experiment_mon.getDistance()
px_per_deg = px_per_cm * cm_per_deg
import pandas as pd

def px_to_deg(px_val):
    """
    Convert from pixels to degrees (based on the 'experiment_mon'
    specifications)
    """
    return px_val / px_per_deg

def make_pp_ellipsis(center_x, center_y, xaxis_r, yaxis_r, ori):
    """
    Form an ellipsis (polygon) stimulus object based on passed
    center coordinates, x/y-axis radii, and rotation angle
    (in degrees, where positive values indicate that the
    axes have been rotated **clockwise**).
    """
    pp_ellipsis = visual.Polygon(
        win=win, name='pp_ellipsis', edges=120,units='deg', 
        size=(xaxis_r*2, yaxis_r*2),
        ori=ori, pos=(center_x, center_y),
        lineWidth=2, lineColor=[0,0,1], lineColorSpace='rgb',
        fillColor=None, fillColorSpace='rgb',
        opacity=1, depth=-10.0, interpolate=True)
    return pp_ellipsis

# dictionary of read instructions/messages participant sound file paths and
# durations
read_audio_info_dicts = {
    'read_intro': {
        'fpath': os.path.join(READ_AUDIO_DIR_PATH, READ_AUDIO_INTRO_FNAME),
        'dur': READ_AUDIO_INTRO_DUR,
    },
    'read_train1_inst': {
        'fpath': os.path.join(READ_AUDIO_DIR_PATH, READ_AUDIO_TRAIN1_INST_FNAME),
        'dur': READ_AUDIO_TRAIN1_INST_DUR,
    },
    'read_train1_corr': {
        'fpath': os.path.join(READ_AUDIO_DIR_PATH, READ_AUDIO_TRAIN1_CORR_FNAME),
        'dur': READ_AUDIO_TRAIN1_CORR_DUR,
    },
    'read_train1_incorr': {
        'fpath': os.path.join(READ_AUDIO_DIR_PATH, READ_AUDIO_TRAIN1_INCORR_FNAME),
        'dur': READ_AUDIO_TRAIN1_INCORR_DUR,
    },
    'read_train2_inst': {
        'fpath': os.path.join(READ_AUDIO_DIR_PATH, READ_AUDIO_TRAIN2_INST_FNAME),
        'dur': READ_AUDIO_TRAIN2_INST_DUR,
    },
    'read_train2_corr': {
        'fpath': os.path.join(READ_AUDIO_DIR_PATH, READ_AUDIO_TRAIN2_CORR_FNAME),
        'dur': READ_AUDIO_TRAIN2_CORR_DUR,
    },
    'read_train2_incorr': {
        'fpath': os.path.join(READ_AUDIO_DIR_PATH, READ_AUDIO_TRAIN2_INCORR_FNAME),
        'dur': READ_AUDIO_TRAIN2_INCORR_DUR,
    },
    'read_start': {
        'fpath': os.path.join(READ_AUDIO_DIR_PATH, READ_AUDIO_START_FNAME),
        'dur': READ_AUDIO_START_DUR,
    }
}
# enable (True) / disable (False) debug mode
# make sure that this is set to **False** before
# testing any participants
DEBUG_ON = False

### BEGIN SET CONSTANTS ###
# experiment constants/settings setup code snippet
# size specifications, EXCEPT THOSE FOR IMAGES, are in degrees.
# duration/time specifications are in seconds, unless otherwise
# specified.

# background color (used by experiment monitor settings)
BG_COL = [1, 1, 1]

# medium text size
TXT_SIZE_M = 1

# neutral text color
TXT_COL_NEUTRAL = [-1, -1, -1]

# regular text wrap width
TXT_WRAP_WIDTH = 28

# 'continue' and 'start' button y coordinate
BTN_Y = -12

# button text size
BTN_TXT_HEIGHT = 1.2

# 'continue' and 'start' button background/fill color
BTN_FILL_COL = [0.5, 0.5, 0.5]
# 'continue' and 'start' button border/line color
BTN_BORDER_COL = [-1, -1, -1]
# button text color
BTN_TXT_COL = BTN_BORDER_COL
# button width/height
BTN_WIDTH = 4
BTN_HEIGHT = 2
# button text height
BTN_TXT_HEIGHT = BTN_HEIGHT * 0.5

# audio (read instructions/messages to participant) volume
AUDIO_VOLUME = 1

## BEGIN INTRO SECTION ##
# intro text y coordinate
INTRO_TXT_Y = 3
## END INTRO SECTION ##

## BEGIN TRAIN/PRACTICE SECTION ##
# training trials image borders color
TRAIN_BORDER_COL = [0,0,0]
#  training trials image borders active (clicks are welcome) color
TRAIN_BORDER_ACTIVE_COL = [-0.5,-0.5,0.5]
# training trial 1 instruction y coordinate
TRAIN1_TXT_INST_Y = 7
# training trial 1 target y coordinate #changed from 3 JN
TRAIN1_TARGET_Y = 2
# training trial 1 context y coordinate #changed from -3 JN
TRAIN1_CONTEXT_Y = -4
# training trial 1 target width/height IN PIXELS
TRAIN1_TARGET_ORIG_WIDTH = 160
TRAIN1_TARGET_ORIG_HEIGHT = 120
# training trial 1 images resize factor
TRAIN1_RESIZE_FACTOR = 1
train1_target_width = TRAIN1_RESIZE_FACTOR * px_to_deg(TRAIN1_TARGET_ORIG_WIDTH)
train1_target_height = TRAIN1_RESIZE_FACTOR * px_to_deg(TRAIN1_TARGET_ORIG_HEIGHT)
# training trial 1 context width/height IN PIXELS
TRAIN1_CONTEXT_ORIG_WIDTH = 430
TRAIN1_CONTEXT_ORIG_HEIGHT = 270
train1_context_width = TRAIN1_RESIZE_FACTOR * px_to_deg(TRAIN1_CONTEXT_ORIG_WIDTH)
train1_context_height = TRAIN1_RESIZE_FACTOR * px_to_deg(TRAIN1_CONTEXT_ORIG_HEIGHT)
# training trial 2 instruction y coordinate
TRAIN2_TXT_INST_Y = 9.5
# training trial 2 target y coordinate
TRAIN2_TARGET_X = -4
TRAIN2_TARGET_Y = -2.5
# training trial 2 context x/y coordinates
TRAIN2_CONTEXT_X = 4
TRAIN2_CONTEXT_Y = -2.5
# training trial 2 target width/height IN PIXELS
TRAIN2_TARGET_ORIG_WIDTH = 119
TRAIN2_TARGET_ORIG_HEIGHT = 133
# training trial 2 images resize factor
TRAIN2_RESIZE_FACTOR = 0.9
train2_target_width = TRAIN2_RESIZE_FACTOR * px_to_deg(TRAIN2_TARGET_ORIG_WIDTH)
train2_target_height = TRAIN2_RESIZE_FACTOR * px_to_deg(TRAIN2_TARGET_ORIG_HEIGHT)
# training trial 2 context width/height IN PIXELS
TRAIN2_CONTEXT_ORIG_WIDTH = 250
TRAIN2_CONTEXT_ORIG_HEIGHT = 600
train2_context_width = TRAIN2_RESIZE_FACTOR * px_to_deg(TRAIN2_CONTEXT_ORIG_WIDTH)
train2_context_height = TRAIN2_RESIZE_FACTOR * px_to_deg(TRAIN2_CONTEXT_ORIG_HEIGHT)

## END TRAIN SECTION ##

## BEGIN PRE-POPOUT INSTRUCTION/TRAIN SECTION ##
# pre-popout trials instruction text y coordinate
PREPOPOUT_TXT_Y = 6

# training pre-popout section images (marked/nonmarked)
TRAIN3_SHOWPOPOUT_X = -8
TRAIN3_SHOWPOPOUT_Y = -6
TRAIN3_SHOWPOPOUT_ORIG_WIDTH = 250
TRAIN3_SHOWPOPOUT_ORIG_HEIGHT = 600
TRAIN3_RESIZE_FACTOR = 0.9
train3_showpopout_width = TRAIN3_RESIZE_FACTOR * px_to_deg(TRAIN3_SHOWPOPOUT_ORIG_WIDTH)
train3_showpopout_height = TRAIN3_RESIZE_FACTOR * px_to_deg(TRAIN3_SHOWPOPOUT_ORIG_HEIGHT)

TRAIN4_SHOWPOPOUT_X = 0
TRAIN4_SHOWPOPOUT_Y = -6
TRAIN4_SHOWPOPOUT_ORIG_WIDTH = 500
TRAIN4_SHOWPOPOUT_ORIG_HEIGHT = 350
TRAIN4_RESIZE_FACTOR = 0.7
train4_showpopout_width = TRAIN4_RESIZE_FACTOR * px_to_deg(TRAIN4_SHOWPOPOUT_ORIG_WIDTH)
train4_showpopout_height = TRAIN4_RESIZE_FACTOR * px_to_deg(TRAIN4_SHOWPOPOUT_ORIG_HEIGHT)

TRAIN5_SHOWPOPOUT_X = 8
TRAIN5_SHOWPOPOUT_Y = -6
TRAIN5_SHOWPOPOUT_ORIG_WIDTH = 500
TRAIN5_SHOWPOPOUT_ORIG_HEIGHT = 350
TRAIN5_RESIZE_FACTOR = 0.7
train5_showpopout_width = TRAIN5_RESIZE_FACTOR * px_to_deg(TRAIN5_SHOWPOPOUT_ORIG_WIDTH)
train5_showpopout_height = TRAIN5_RESIZE_FACTOR * px_to_deg(TRAIN5_SHOWPOPOUT_ORIG_HEIGHT)

# duration of trial pre-popout instruction text
INST_SHOWPOPOUT_TXT_DUR = 40
# duration for which non-marked versions of example images
# will be shown
INST_SHOWPOPOUT_NONMARKED_DUR = 20
# duration for which marked versions of example images
# will be shown
INST_SHOWPOPOUT_MARKED_DUR = 20


## END PRE-POPOUT INSTRUCTION/TRAIN SECTION ##

## BEGIN TRIAL SECTION ##
# duration of pre-trial instruction message.
# duration here is specified in
# **number of frames** ie on a standard setup you will
# specify 60 (frames) for 1s, since most computers
# will show 60 frames/s
TRIALINST_DUR = 3 * 60
# letter height of pre-trial instruction message
TRIALINST_TXT_HEIGHT = 3

if not DEBUG_ON:
    # per-target maximum duration of multi-target (type 'once')
    # trials
    PER_TARGET_MAX_DUR = 30
    # maximum duration of single-target ('heap' and 'repeat' trials)
    SINGLE_TARGET_DUR = 90
    # maximum duration of single-target 'popout' (non-embedded)
    # trials
    SINGLE_TARGET_DUR_POPOUT = 30
else:
    PER_TARGET_MAX_DUR = 5
    SINGLE_TARGET_DUR = 5
    SINGLE_TARGET_DUR_POPOUT = 5
# duration of 'blocking period' after a new target is
# presented, during which the participant cannot click
BLOCKING_DUR = 0.4

# trial target width/height IN PIXELS
TRIAL_TARGET_ORIG_WIDTH = 210
TRIAL_TARGET_ORIG_HEIGHT = 180
# trial target resize factor
TRIAL_TARGET_RESIZE_FACTOR = 1
trial_target_width = TRIAL_TARGET_RESIZE_FACTOR * px_to_deg(TRIAL_TARGET_ORIG_WIDTH)
trial_target_height = TRIAL_TARGET_RESIZE_FACTOR * px_to_deg(TRIAL_TARGET_ORIG_HEIGHT)
# trial target y coordinate
TRIAL_TARGET_Y = 11
# trial context maximum width/height IN DEGREES
# (each image here has a different size from the others)
TRIAL_CONTEXT_MAX_WIDTH = 35
TRIAL_CONTEXT_MAX_HEIGHT = 20
# trial context y coordinate
TRIAL_CONTEXT_Y = -3

# for 'once' type trials, duration of inter-target-interval
# (ie pause after each target being presented before next
# target is shown/trial ends). duration here is specified in
# **number of frames** ie on a standard setup you will
# specify 60 (frames) for 1s, since most computers
# will show 60 frames/s
ONCE_INTER_TARGET_DUR = 60

## END TRIAL SECTION ##

### END SET CONSTANTS ###
def get_resize_factor(orig_w, orig_h, max_w, max_h):
    """
    Returns a resize factor based on former width/height and maximum
    allowed width/height.
    """
    if ((orig_h / max_h) > (orig_w / max_w)):
        return max_h/orig_h
    else:
        return max_w/orig_w

# note that this routine, for showing popout instructions
# and changing trial/target durations,
# is set to only be shown/run when 6 trials have passed,
# since trials 7 and onwards are 'popout' trials. the condtional
# logic for skipping this routine otherwise is implemented
# using conditional logic for the 'nReps' parameter of the enclosing
# loop, as suggested on the PsychoPy forums:
# https://discourse.psychopy.org/t/implementing-conditional-routines/15193/2



# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_thisDir)
# Store info about the experiment session
psychopyVersion = '2022.1.2'
expName = 'disembedding_task'  # from the Builder filename that created this script
expInfo = {'participant': '', 'order_group_number': ''}
dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName)
if dlg.OK == False:
    core.quit()  # user pressed cancel
expInfo['date'] = data.getDateStr()  # add a simple timestamp
expInfo['expName'] = expName
expInfo['psychopyVersion'] = psychopyVersion

# Data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
filename = _thisDir + os.sep + u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])

# An ExperimentHandler isn't essential but helps with data saving
thisExp = data.ExperimentHandler(name=expName, version='',
    extraInfo=expInfo, runtimeInfo=None,
    originPath='/Users/workingman/Documents/ASD_and_Synesthesia/Python/psychopy_shared_github/disembedding_task/disembedding_task.py',
    savePickle=True, saveWideText=True,
    dataFileName=filename)
# save a log file for detail verbose info
logFile = logging.LogFile(filename+'.log', level=logging.DEBUG)
logging.console.setLevel(logging.WARNING)  # this outputs to the screen, not a file

endExpNow = False  # flag for 'escape' or other condition => quit the exp
frameTolerance = 0.001  # how close to onset before 'same' frame

# Start Code - component code to be run after the window creation

# Setup the Window
win = visual.Window(
    size=[1280, 750], fullscr=False, screen=0, 
    winType='pyglet', allowGUI=True, allowStencil=False,
    monitor=experiment_mon, color=BG_COL, colorSpace='rgb',
    blendMode='avg', useFBO=True, 
    units='deg')
# store frame rate of monitor if we can measure it
expInfo['frameRate'] = win.getActualFrameRate()
if expInfo['frameRate'] != None:
    frameDur = 1.0 / round(expInfo['frameRate'])
else:
    frameDur = 1.0 / 60.0  # could not measure, so guess
# Setup ioHub
ioConfig = {}
ioSession = ioServer = eyetracker = None

# create a default keyboard (e.g. to check for escape)
defaultKeyboard = keyboard.Keyboard(backend='ptb')

# Initialize components for Routine "setup"
setupClock = core.Clock()
# read in target ellipses data
ellipses_data_path = os.path.join('stimuli', 'stimuli_roi_coordinates.csv')
ellipses_df = pd.read_csv(ellipses_data_path)

# list that will be updated for each trial with that
# trial's ellipses stimuli objects
trial_ellipses = []

# mouse for registering participant clicks, with
# related attributes and variables
mouse = event.Mouse(win=win)
x, y = [None, None]
mouse.mouseClock = core.Clock()
prevButtonState = []
mouse.x = []
mouse.y = []
mouse.leftButton = []
mouse.midButton = []
mouse.rightButton = []
mouse.time = []

# continue button/text
rectangle_continue = visual.Rect(
    win=win, name='rectangle_intro_continue',units='deg', 
    width=(BTN_WIDTH*1.5, BTN_HEIGHT)[0], height=(BTN_WIDTH*1.5, BTN_HEIGHT)[1],
    ori=0, pos=(0, BTN_Y),
    lineWidth=2, lineColor=BTN_BORDER_COL, lineColorSpace='rgb',
    fillColor=BTN_FILL_COL, fillColorSpace='rgb',
    opacity=1, depth=-4.0, interpolate=True)
text_continue = visual.TextStim(win=win, name='text_intro_continue',
    text=CONTINUE_TXT,
    font='Arial',
    units='deg', pos=(0, BTN_Y), height=BTN_TXT_HEIGHT, wrapWidth=None, ori=0, 
    color=BTN_TXT_COL, colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-6.0);

# cross for marking where participant has clicked
cross_stimulus = visual.ShapeStim(
    win, 
    pos=(0, 0), 
    depth=-2, 
    vertices='cross', 
    size=(0.2, 0.2), 
    lineColor=[-0.8, -0.8, 1], 
    fillColor=[-0.8, -0.8, 1],
    units='deg')
# list for keeping track of where cross should be
# drawn (ie where participant has clicked during
# current trial)
cross_position_list = []

# put together dictionary of sound stimuli for read instructions/messages
read_audio_sound_dict = {}

for info_dict_key in read_audio_info_dicts:
    info_dict = read_audio_info_dicts[info_dict_key]
    fpath = info_dict['fpath']
    dur = info_dict['dur']
    read_audio_sound_dict[info_dict_key] = {
        'dur': dur,
        'sound': sound.Sound(
            value=fpath, 
            secs=dur,
            stereo=True,
            hamming=True,
            name=info_dict_key,
            volume=AUDIO_VOLUME
        )
    }


# Initialize components for Routine "intro"
introClock = core.Clock()
text_intro = visual.TextStim(win=win, name='text_intro',
    text=INTRO_TXT,
    font='Arial',
    units='deg', pos=(0, 0), height=TXT_SIZE_M, wrapWidth=TXT_WRAP_WIDTH, ori=0, 
    color=TXT_COL_NEUTRAL, colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
mouse_intro = event.Mouse(win=win)
x, y = [None, None]
mouse_intro.mouseClock = core.Clock()
sound_read_audio_intro = sound.Sound(os.path.join(READ_AUDIO_DIR_PATH, READ_AUDIO_INTRO_FNAME), secs=-1, stereo=True, hamming=True,
    name='sound_read_audio_intro')
sound_read_audio_intro.setVolume(AUDIO_VOLUME)

# Initialize components for Routine "train_1"
train_1Clock = core.Clock()
# these stimuli must be manually created to control when they are drawn,
# to avoid them overlaying (through autodraw) 'correct' circles that are to 
# be drawn on top
image_train_target = visual.ImageStim(
    win=win,
    name='image_train_target', 
    image=os.path.join("stimuli", "train", "train1_target_new.jpg"), mask=None,
    ori=0, pos=(0, TRAIN1_TARGET_Y), size=(train1_target_width, train1_target_height),
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=512, interpolate=True, depth=-2.0)

image_train_context = visual.ImageStim(
    win=win,
    name='image_train_context', 
    image=os.path.join("stimuli", "train", "train1_context_new.jpg"), mask=None,
    ori=0, pos=(0, TRAIN1_CONTEXT_Y), size=(train1_context_width, train1_context_height),
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=512, interpolate=True, depth=-3.0)

# train target/context border stimuli
rect_train_target = visual.Rect(
    win=win, name='rect_train_target',units='deg', 
    size=(train1_target_width, train1_target_height),
    ori=0, pos=(0, TRAIN1_TARGET_Y),
    lineWidth=3, lineColor=TRAIN_BORDER_COL, lineColorSpace='rgb',
    fillColor=None, fillColorSpace='rgb',
    opacity=1, depth=-2.0, interpolate=True)

rect_train_context = visual.Rect(
    win=win, name='rect_train_context',units='deg', 
    size=(train1_context_width, train1_context_height),
    ori=0, pos=(0, TRAIN1_CONTEXT_Y),
    lineWidth=3, lineColor=TRAIN_BORDER_COL, lineColorSpace='rgb',
    fillColor=None, fillColorSpace='rgb',
    opacity=1, depth=-2.0, interpolate=True)

text_train1_inst = visual.TextStim(win=win, name='text_train1_inst',
    text=TRAIN1_TXT_INST,
    font='Arial',
    units='deg', pos=(0, TRAIN1_TXT_INST_Y), height=TXT_SIZE_M, wrapWidth=TXT_WRAP_WIDTH, ori=0, 
    color=TXT_COL_NEUTRAL, colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-1.0);

# Initialize components for Routine "train_2"
train_2Clock = core.Clock()
text_train2_inst = visual.TextStim(win=win, name='text_train2_inst',
    text=TRAIN2_TXT_INST,
    font='Arial',
    units='deg', pos=(0, TRAIN2_TXT_INST_Y), height=TXT_SIZE_M, wrapWidth=TXT_WRAP_WIDTH, ori=0, 
    color=TXT_COL_NEUTRAL, colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-1.0);

# Initialize components for Routine "start"
startClock = core.Clock()
text_start = visual.TextStim(win=win, name='text_start',
    text=START_TXT_INST,
    font='Arial',
    units='deg', pos=(0, 0), height=TXT_SIZE_M, wrapWidth=TXT_WRAP_WIDTH, ori=0, 
    color=TXT_COL_NEUTRAL, colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
mouse_start = event.Mouse(win=win)
x, y = [None, None]
mouse_start.mouseClock = core.Clock()

# Initialize components for Routine "trial_instruction"
trial_instructionClock = core.Clock()
text_trialinstruction = visual.TextStim(win=win, name='text_trialinstruction',
    text='',
    font='Arial',
    units='deg', pos=(0, 0), height=TRIALINST_TXT_HEIGHT, wrapWidth=TXT_WRAP_WIDTH, ori=0, 
    color=TXT_COL_NEUTRAL, colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
# specify path to where context image sizes specification is located
context_img_spec_path = os.path.join('stimuli', 'context_img_sizes.csv')

context_img_df = pd.read_csv(context_img_spec_path)

# form a dictionary of context images, where image filenames are the
# keys
context_img_dict = {}
for i in range(len(context_img_df)):
    row = context_img_df.iloc[i]
    fname = row['filename']
    stim_name = fname.replace('.csv', '')
    orig_w_deg = px_to_deg(row['width_px'])
    orig_h_deg = px_to_deg(row['height_px'])
    context_resize_factor = get_resize_factor(
        orig_w_deg, 
        orig_h_deg,
        TRIAL_CONTEXT_MAX_WIDTH,
        TRIAL_CONTEXT_MAX_HEIGHT
    )
    set_w_deg = orig_w_deg * context_resize_factor
    set_h_deg = orig_h_deg * context_resize_factor
    context_img_dict[fname] = {
        'stim': visual.ImageStim(
            win=win,
            name= f'image_{stim_name}_target', 
            image=os.path.join("stimuli", "trial", "context", fname), 
            mask=None,
            ori=0, 
            pos=(0, TRIAL_CONTEXT_Y), 
            size=(set_w_deg, set_h_deg),
            color=[1,1,1], 
            colorSpace='rgb', 
            opacity=1,
            flipHoriz=False, 
            flipVert=False,
            texRes=512, 
            interpolate=True, 
            depth=-2.0
        ),
        'resize_factor': context_resize_factor
    }

# form dictionary of target images
target_dir_path = os.path.join('stimuli', 'trial', 'target')
target_fnames = [x for x in os.listdir(target_dir_path) if not x.startswith('.') and x.endswith('.jpg')]
target_img_dict = {}
for fname in target_fnames:
    stim_name = fname.replace('.csv', '')
    target_img_dict[fname]  = visual.ImageStim(
        win=win,
        name= f'image_{stim_name}_target', 
        image=os.path.join(target_dir_path, fname), 
        mask=None,
        ori=0, 
        pos=(0, TRIAL_TARGET_Y), 
        size=(trial_target_width, trial_target_height),
        color=[1,1,1], 
        colorSpace='rgb', 
        opacity=1,
        flipHoriz=False, 
        flipVert=False,
        texRes=512, 
        interpolate=True, 
        depth=-2.0
    )

# get trial stimuli/order specifications for this participant, based on group, and
# put in a list of dictionaries
group_no = expInfo['order_group_number']
if not group_no.isnumeric() or int(float(group_no)) not in range(1, 5):
    err_mess = 'Invalid group number! Must be a single digit between 1-4. Please try again.'
    print(err_mess)
    raise IndexError(err_mess)
group_no = int(float(group_no))
spec_path = os.path.join('order_specifications', f'trial_stim_specifications_{group_no}.csv')
trial_df = pd.read_csv(spec_path)
trial_ls = []
for i in range(len(trial_df)):
    row = trial_df.iloc[i]
    trial_dict = {
        'context_fname': row['context_filename'],
        'target_fnames': eval(row['target_filenames']),
        'type': row['type']
    }
    trial_ls.append(trial_dict)


# border stimuli
rect_trial_target = visual.Rect(
    win=win, name='rect_trial_target',units='deg', 
    size=(trial_target_width, trial_target_height),
    ori=0, pos=(0, TRIAL_TARGET_Y),
    lineWidth=2, lineColor=[1,1,1], lineColorSpace='rgb',
    fillColor=None, fillColorSpace='rgb',
    opacity=1, depth=-2.0, interpolate=True)

rect_trial_context = visual.Rect(
    win=win, name='rect_trial_context',units='deg', 
    size=(TRIAL_CONTEXT_MAX_WIDTH, TRIAL_CONTEXT_MAX_HEIGHT),
    ori=0, pos=(0, TRIAL_CONTEXT_Y),
    lineWidth=2, lineColor=[-1,-1,-1], lineColorSpace='rgb',
    fillColor=None, fillColorSpace='rgb',
    opacity=1, depth=-2.0, interpolate=True)

# initialize trial counter
trial_counter = 0

# dictionary pairing trial types and which message to
# show
type_mess_dict = {
    'once': ONCE_TXT,
    'repeat': REP_TXT,
    'heap': REP_TXT
}

# initialize trial instruction message variable (to prevent undefined variable error)
trial_type = trial_ls[trial_counter]['type']
trialinstruction_mess = type_mess_dict[trial_type]


# Initialize components for Routine "trial"
trialClock = core.Clock()
text_trial_timekeeper = visual.TextStim(win=win, name='text_trial_timekeeper',
    text=None,
    font='Arial',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-1.0);

# Initialize components for Routine "popout_instruction"
popout_instructionClock = core.Clock()
text_popoutinst = visual.TextStim(win=win, name='text_popoutinst',
    text=POPOUT_TXT,
    font='Arial',
    units='deg', pos=(0, PREPOPOUT_TXT_Y), height=TXT_SIZE_M, wrapWidth=TXT_WRAP_WIDTH, ori=0, 
    color=TXT_COL_NEUTRAL, colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
image_showpopout1_nonmarked = visual.ImageStim(
    win=win,
    name='image_showpopout1_nonmarked', units='deg', 
    image='stimuli/train/train3_showpopout_nonmarked.jpg', mask=None, anchor='center',
    ori=0, pos=(TRAIN3_SHOWPOPOUT_X, TRAIN3_SHOWPOPOUT_Y), size=(train3_showpopout_width, train3_showpopout_height),
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=512, interpolate=True, depth=-2.0)
image_showpopout1_marked = visual.ImageStim(
    win=win,
    name='image_showpopout1_marked', units='deg', 
    image='stimuli/train/train3_showpopout_marked.jpg', mask=None, anchor='center',
    ori=0, pos=(TRAIN3_SHOWPOPOUT_X, TRAIN3_SHOWPOPOUT_Y), size=(train3_showpopout_width, train3_showpopout_height),
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=512, interpolate=True, depth=-3.0)
sound_popoutinst_readinst = sound.Sound(os.path.join(READ_AUDIO_DIR_PATH, READ_AUDIO_INTRO_FNAME), secs=-1, stereo=True, hamming=True,
    name='sound_popoutinst_readinst')
sound_popoutinst_readinst.setVolume(1)
image_showpopout2_nonmarked = visual.ImageStim(
    win=win,
    name='image_showpopout2_nonmarked', units='deg', 
    image='stimuli/train/train4_showpopout_nonmarked.jpg', mask=None, anchor='center',
    ori=0, pos=(TRAIN4_SHOWPOPOUT_X, TRAIN4_SHOWPOPOUT_Y), size=(train4_showpopout_width, train4_showpopout_height),
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=512, interpolate=True, depth=-5.0)
image_showpopout2_marked = visual.ImageStim(
    win=win,
    name='image_showpopout2_marked', units='deg', 
    image='stimuli/train/train4_showpopout_marked.jpg', mask=None, anchor='center',
    ori=0, pos=(TRAIN4_SHOWPOPOUT_X, TRAIN4_SHOWPOPOUT_Y), size=(train4_showpopout_width, train4_showpopout_height),
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=512, interpolate=True, depth=-6.0)
image_showpopout3_nonmarked = visual.ImageStim(
    win=win,
    name='image_showpopout3_nonmarked', units='deg', 
    image='stimuli/train/train5_showpopout_nonmarked.jpg', mask=None, anchor='center',
    ori=0, pos=(TRAIN5_SHOWPOPOUT_X, TRAIN5_SHOWPOPOUT_Y), size=(train5_showpopout_width, train5_showpopout_height),
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=512, interpolate=True, depth=-7.0)
image_showpopout3_marked = visual.ImageStim(
    win=win,
    name='image_showpopout3_marked', units='deg', 
    image='stimuli/train/train5_showpopout_marked.jpg', mask=None, anchor='center',
    ori=0, pos=(TRAIN5_SHOWPOPOUT_X, TRAIN5_SHOWPOPOUT_Y), size=(train5_showpopout_width, train5_showpopout_height),
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=512, interpolate=True, depth=-8.0)

# Initialize components for Routine "end"
endClock = core.Clock()
text_end = visual.TextStim(win=win, name='text_end',
    text=END_TXT,
    font='Arial',
    units='deg', pos=(0, 0), height=TXT_SIZE_M, wrapWidth=TXT_WRAP_WIDTH, ori=0, 
    color=TXT_COL_NEUTRAL, colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);

# Create some handy timers
globalClock = core.Clock()  # to track the time since experiment started
routineTimer = core.CountdownTimer()  # to track time remaining of each (non-slip) routine 

# ------Prepare to start Routine "setup"-------
continueRoutine = True
# update component parameters for each repeat
# keep track of which components have finished
setupComponents = []
for thisComponent in setupComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
setupClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "setup"-------
while continueRoutine:
    # get current time
    t = setupClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=setupClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in setupComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "setup"-------
for thisComponent in setupComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
# the Routine "setup" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# ------Prepare to start Routine "intro"-------
continueRoutine = True
# update component parameters for each repeat
# setup some python lists for storing info about the mouse_intro
mouse_intro.clicked_name = []
gotValidClick = False  # until a click is received
# fetch audio of read instructions and its duration
read_sound = read_audio_sound_dict['read_intro']['sound']
read_dur = read_audio_sound_dict['read_intro']['dur']
# start playing instructions
read_sound.play()

sound_read_audio_intro.setSound(os.path.join(READ_AUDIO_DIR_PATH, READ_AUDIO_INTRO_FNAME), secs=READ_AUDIO_INTRO_DUR, hamming=True)
sound_read_audio_intro.setVolume(AUDIO_VOLUME, log=False)
# keep track of which components have finished
introComponents = [text_intro, mouse_intro, sound_read_audio_intro]
for thisComponent in introComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
introClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "intro"-------
while continueRoutine:
    # get current time
    t = introClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=introClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *text_intro* updates
    if text_intro.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        text_intro.frameNStart = frameN  # exact frame index
        text_intro.tStart = t  # local t and not account for scr refresh
        text_intro.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(text_intro, 'tStartRefresh')  # time at next scr refresh
        text_intro.setAutoDraw(True)
    # *mouse_intro* updates
    if mouse_intro.status == NOT_STARTED and t >= read_audio_sound_dict['read_intro']['dur']-frameTolerance:
        # keep track of start time/frame for later
        mouse_intro.frameNStart = frameN  # exact frame index
        mouse_intro.tStart = t  # local t and not account for scr refresh
        mouse_intro.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(mouse_intro, 'tStartRefresh')  # time at next scr refresh
        mouse_intro.status = STARTED
        mouse_intro.mouseClock.reset()
        prevButtonState = mouse_intro.getPressed()  # if button is down already this ISN'T a new click
    if mouse_intro.status == STARTED:  # only update if started and not finished!
        buttons = mouse_intro.getPressed()
        if buttons != prevButtonState:  # button state changed?
            prevButtonState = buttons
            if sum(buttons) > 0:  # state changed to a new click
                # check if the mouse was inside our 'clickable' objects
                gotValidClick = False
                try:
                    iter(rectangle_continue)
                    clickableList = rectangle_continue
                except:
                    clickableList = [rectangle_continue]
                for obj in clickableList:
                    if obj.contains(mouse_intro):
                        gotValidClick = True
                        mouse_intro.clicked_name.append(obj.name)
                if gotValidClick:  
                    continueRoutine = False  # abort routine on response
    # if read instructions audio has finished playing
    if tThisFlip >= read_dur:
        rectangle_continue.draw()
        text_continue.draw()
    
    # start/stop sound_read_audio_intro
    if sound_read_audio_intro.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        sound_read_audio_intro.frameNStart = frameN  # exact frame index
        sound_read_audio_intro.tStart = t  # local t and not account for scr refresh
        sound_read_audio_intro.tStartRefresh = tThisFlipGlobal  # on global time
        sound_read_audio_intro.play(when=win)  # sync with win flip
    if sound_read_audio_intro.status == STARTED:
        # is it time to stop? (based on global clock, using actual start)
        if tThisFlipGlobal > sound_read_audio_intro.tStartRefresh + READ_AUDIO_INTRO_DUR-frameTolerance:
            # keep track of stop time/frame for later
            sound_read_audio_intro.tStop = t  # not accounting for scr refresh
            sound_read_audio_intro.frameNStop = frameN  # exact frame index
            win.timeOnFlip(sound_read_audio_intro, 'tStopRefresh')  # time at next scr refresh
            sound_read_audio_intro.stop()
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in introComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "intro"-------
for thisComponent in introComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
# store data for thisExp (ExperimentHandler)
thisExp.nextEntry()
sound_read_audio_intro.stop()  # ensure sound has stopped at end of routine
# the Routine "intro" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# ------Prepare to start Routine "train_1"-------
continueRoutine = True
# update component parameters for each repeat
# reset mouse save data attributes
mouse.x = []
mouse.y = []
mouse.time = []

# reset counter of number of correct responses
correct_counter = 0

# reset variable indicating if participant has clicked
has_clicked = False

# reset variable indicating if routine is ready for showing
# continue button
ready_continue = False

# reset continue button
rectangle_continue.opacity = 0
# empty continue text component's text, which is filled
# in once participant is allowed to continue
# (can't simply change opacity of thext text because of a PsychoPy bug)
text_continue.text = ""

# reset list of cross drawing positions
cross_position_list.clear()

# clear the list of ellipses stimuli, to make it ready for this trial
trial_ellipses.clear()
# specify the resize factor for this routine
resize_factor = TRAIN1_RESIZE_FACTOR
# specify the context y coordinate offset
context_y = TRAIN1_CONTEXT_Y
# specify the context image's filename (for filtering dataframe rows
# below)
context_filename = 'train1_context.jpg'

# find which row indices in the ellipses_df that correspond to this trial's
# target ellipses data
df_indices = list(ellipses_df[ellipses_df['filename'] == context_filename].index)
for index in df_indices:
    center_x = px_to_deg(ellipses_df.loc[index, 'center_x']) * resize_factor
    center_y = px_to_deg(ellipses_df.loc[index, 'center_y']) * resize_factor + context_y
    xaxis_r = px_to_deg(ellipses_df.loc[index, 'xaxis_r']) * resize_factor
    yaxis_r = px_to_deg(ellipses_df.loc[index, 'yaxis_r']) * resize_factor
    orientation = ellipses_df.loc[index, 'orientation']
    new_ellipsis = make_pp_ellipsis(
        center_x, 
        center_y, 
        xaxis_r, 
        yaxis_r, 
        orientation
    )
    trial_ellipses.append(new_ellipsis)

# fetch audio of read instructions and its duration
read_sound_inst = read_audio_sound_dict['read_train1_inst']['sound']
read_dur_inst = read_audio_sound_dict['read_train1_inst']['dur']
# fetch audio of incorrect/correct messages and durations
read_sound_corr = read_audio_sound_dict['read_train1_corr']['sound']
read_dur_corr = read_audio_sound_dict['read_train1_corr']['dur']
read_sound_incorr = read_audio_sound_dict['read_train1_incorr']['sound']
read_dur_incorr = read_audio_sound_dict['read_train1_incorr']['dur']
# reset indicator of whether borders have switched to 'active' (responses
# welcome) color
active_borders = False
# reset indicator of whether feedback has started,
# and variable indicating start time of feedback
feedback_started = False
feedback_start_time = 9999
# start playing instructions
read_sound_inst.play()

# keep track of which components have finished
train_1Components = [text_train1_inst]
for thisComponent in train_1Components:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
train_1Clock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "train_1"-------
while continueRoutine:
    # get current time
    t = train_1Clock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=train_1Clock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    image_train_target.draw()
    image_train_context.draw()
    rect_train_target.draw()
    rect_train_context.draw()
    rectangle_continue.draw()
    text_continue.draw()
    
    # instructions audio has finished?
    if tThisFlip >= read_dur_inst:
        if not active_borders:
            rect_train_target.lineColor = TRAIN_BORDER_ACTIVE_COL
            rect_train_context.lineColor = TRAIN_BORDER_ACTIVE_COL
            active_borders = True
        buttons = mouse.getPressed()
        if buttons != prevButtonState:  # button state changed?
            prevButtonState = buttons
            if sum(buttons) > 0:  # state changed to a new click
                x, y = mouse.getPos()
                mouse.x.append(x)
                mouse.y.append(y)
                # if continue button is already shown and trial is otherwise
                # over
                if ready_continue:
                    if rectangle_continue.contains(mouse):
                        continueRoutine = False
                elif image_train_context.contains(mouse): 
                    has_clicked = True
                    cross_position_list.append((x, y))
                    # go through all ellipses and check for each one if mouse
                    # was clicked within it
                    for ell_ind in range(len(trial_ellipses)):
                        if trial_ellipses[ell_ind].contains(mouse):
                            correct_counter += 1
                buttons = mouse.getPressed()
                mouse.time.append(mouse.mouseClock.getTime())
    
    if has_clicked:
        for ellipsis in trial_ellipses:
            ellipsis.draw()
        if not feedback_started:
            if correct_counter:
                # set feedback sound and duration
                feedback_sound = read_sound_corr
                feedback_dur = read_dur_corr
                # set top text to feedback text
                text_train1_inst.text = TRAIN1_TXT_CORR
            else:
                feedback_sound = read_sound_incorr
                feedback_dur = read_dur_incorr
                text_train1_inst.text = TRAIN1_TXT_INCORR
            feedback_sound.play()
            feedback_start_time = t
            feedback_started = True
        feedback_finished = (t - feedback_start_time) > feedback_dur
        if not ready_continue and feedback_finished:
            ready_continue = True
            # show continue button
            rectangle_continue.opacity = 1
            text_continue.text = CONTINUE_TXT
    
    # draw crosses where participant has clicked
    for position in cross_position_list:
        cross_stimulus.pos = position
        cross_stimulus.draw() 
    
    
    # *text_train1_inst* updates
    if text_train1_inst.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        text_train1_inst.frameNStart = frameN  # exact frame index
        text_train1_inst.tStart = t  # local t and not account for scr refresh
        text_train1_inst.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(text_train1_inst, 'tStartRefresh')  # time at next scr refresh
        text_train1_inst.setAutoDraw(True)
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in train_1Components:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "train_1"-------
for thisComponent in train_1Components:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
# save practice trial data
thisExp.addData('mouse.x', mouse.x)
thisExp.addData('mouse.y', mouse.y)
thisExp.addData('mouse.time', mouse.time)
if correct_counter == 1:
    thisExp.addData('practice_cube_success', 'success')
else:
    thisExp.addData('practice_cube_success', 'failure')
thisExp.nextEntry()
# the Routine "train_1" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# ------Prepare to start Routine "train_2"-------
continueRoutine = True
# update component parameters for each repeat
# reset counter of number of correct responses
correct_counter = 0

# reset click counter
click_counter = 0

# reset mouse save data attributes
mouse.x = []
mouse.y = []
mouse.time = []

# reset variable indicating if routine is ready for showing
# continue button
ready_continue = False

# reset continue button
rectangle_continue.opacity = 0
# empty continue text component's text, which is filled
# in once participant is allowed to continue
# (can't simply change opacity of thext text because of a PsychoPy bug)
text_continue.text = ""

# reset list of cross drawing positions
cross_position_list.clear()

# clear the list of ellipses stimuli, to make it ready for this trial
trial_ellipses.clear()
# specify the resize factor for this routine
resize_factor = TRAIN2_RESIZE_FACTOR
# specify the context x/y coordinate offsets
context_x = TRAIN2_CONTEXT_X
context_y = TRAIN2_CONTEXT_Y
# specify the target/context images' filenames (for filtering dataframe rows
# below, and updating image stimuli)
target_filename = "train2_target.jpg"
context_filename = 'train2_context.jpg'

# find which row indices in the ellipses_df that correspond to this trial's
# target ellipses data
df_indices = list(ellipses_df[ellipses_df['filename'] == context_filename].index)
for index in df_indices:
    center_x = px_to_deg(ellipses_df.loc[index, 'center_x']) * resize_factor + context_x
    center_y = px_to_deg(ellipses_df.loc[index, 'center_y']) * resize_factor + context_y
    xaxis_r = px_to_deg(ellipses_df.loc[index, 'xaxis_r']) * resize_factor
    yaxis_r = px_to_deg(ellipses_df.loc[index, 'yaxis_r']) * resize_factor
    orientation = ellipses_df.loc[index, 'orientation']
    new_ellipsis = make_pp_ellipsis(
        center_x, 
        center_y, 
        xaxis_r, 
        yaxis_r, 
        orientation
    )
    new_ellipsis.correct = ellipses_df.loc[index, 'correct']
    if not new_ellipsis.correct:
        new_ellipsis.lineColor = [1,0,0]
    trial_ellipses.append(new_ellipsis)

# make a shallow copy of ellipses list, from which ellipses
# will be plucked as soon as they are clicked
nonclicked_ellipses = [ell for ell in trial_ellipses if ell.correct]

# update image stimuli
image_train_target.image = os.path.join("stimuli", "train", target_filename)
image_train_target.pos=(TRAIN2_TARGET_X, TRAIN2_TARGET_Y)
image_train_target.size=(train2_target_width, train2_target_height)
image_train_context.image = os.path.join("stimuli", "train", context_filename)
image_train_context.pos = (context_x, context_y)
image_train_context.size=(train2_context_width, train2_context_height)

# update border stimuli
rect_train_target.size = (train2_target_width, train2_target_height)
rect_train_context.size = (train2_context_width, train2_context_height)
rect_train_target.pos=(TRAIN2_TARGET_X, TRAIN2_TARGET_Y)
rect_train_context.pos=(context_x, context_y)


# fetch audio of read instructions and its duration
read_sound_inst = read_audio_sound_dict['read_train2_inst']['sound']
read_dur_inst = read_audio_sound_dict['read_train2_inst']['dur']
# fetch audio of incorrect/correct messages and durations
read_sound_corr = read_audio_sound_dict['read_train2_corr']['sound']
read_dur_corr = read_audio_sound_dict['read_train2_corr']['dur']
read_sound_incorr = read_audio_sound_dict['read_train2_incorr']['sound']
read_dur_incorr = read_audio_sound_dict['read_train2_incorr']['dur']
# reset indicator of whether borders have switched to 'active' (responses
# welcome) color
active_borders = False
# reset indicator of whether feedback has started,
# and variable indicating start time of feedback
feedback_started = False
feedback_start_time = 9999
# start playing instructions
read_sound_inst.play()
# reset border colors to 'neutral'
rect_train_target.lineColor = TRAIN_BORDER_COL
rect_train_context.lineColor = TRAIN_BORDER_COL
# keep track of which components have finished
train_2Components = [text_train2_inst]
for thisComponent in train_2Components:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
train_2Clock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "train_2"-------
while continueRoutine:
    # get current time
    t = train_2Clock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=train_2Clock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    image_train_target.draw()
    image_train_context.draw()
    rect_train_target.draw()
    rect_train_context.draw()
    rectangle_continue.draw()
    text_continue.draw()
    
    # instructions audio has finished?
    if tThisFlip >= read_dur_inst:
        if not active_borders:
            rect_train_target.lineColor = TRAIN_BORDER_ACTIVE_COL
            rect_train_context.lineColor = TRAIN_BORDER_ACTIVE_COL
            active_borders = True
        buttons = mouse.getPressed()
        if buttons != prevButtonState:  # button state changed?
            prevButtonState = buttons
            if sum(buttons) > 0:  # state changed to a new click
                x, y = mouse.getPos()
                mouse.x.append(x)
                mouse.y.append(y)
                # if continue button is already shown and trial is otherwise
                # over
                if ready_continue:
                    if rectangle_continue.contains(mouse):
                        continueRoutine = False
                # if click within context image area
                elif image_train_context.contains(mouse): 
                    click_counter += 1
                    # list of indices of ellipses to be removed
                    rm_ell_inds = []
                    cross_position_list.append((x, y))
                    # go through all ellipses and check for each one if mouse
                    # was clicked within it
                    for ell_ind in range(len(nonclicked_ellipses)):
                        if nonclicked_ellipses[ell_ind].contains(mouse):
                            correct_counter += 1
                            rm_ell_inds.append(ell_ind)
                    # remove clicked ellipses
                    for rm_ell_ind in rm_ell_inds:
                        nonclicked_ellipses.pop(rm_ell_ind)
                        # decrement all 'remove indices' that are higher
                        # than the index whose element was just removed
                        rm_ell_inds = [x-1 if x>rm_ell_ind else x for x in rm_ell_inds]
                buttons = mouse.getPressed()
                mouse.time.append(mouse.mouseClock.getTime())
    
    # checking here against `len(trial_ellipses)-1` since
    # one of the ellipses in this practice trial is incorrect
    if correct_counter >= len(trial_ellipses)-1:
        for ellipsis in trial_ellipses:
            ellipsis.draw()
        if not feedback_started:
            if correct_counter == click_counter:
                # set feedback sound and duration
                feedback_sound = read_sound_corr
                feedback_dur = read_dur_corr
                # set top text to feedback text
                text_train2_inst.text = TRAIN2_TXT_CORR
            else:
                feedback_sound = read_sound_incorr
                feedback_dur = read_dur_incorr
                text_train2_inst.text = TRAIN2_TXT_INCORR
            feedback_sound.play()
            feedback_start_time = t
            feedback_started = True
        feedback_finished = (t - feedback_start_time) > feedback_dur
        if not ready_continue and feedback_finished:
            ready_continue = True
            # show continue button
            rectangle_continue.opacity = 1
            text_continue.text = CONTINUE_TXT
    
    # draw crosses where participant has clicked
    for position in cross_position_list:
        cross_stimulus.pos = position
        cross_stimulus.draw() 
    
    
    # *text_train2_inst* updates
    if text_train2_inst.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        text_train2_inst.frameNStart = frameN  # exact frame index
        text_train2_inst.tStart = t  # local t and not account for scr refresh
        text_train2_inst.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(text_train2_inst, 'tStartRefresh')  # time at next scr refresh
        text_train2_inst.setAutoDraw(True)
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in train_2Components:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "train_2"-------
for thisComponent in train_2Components:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
# save practice trial data
thisExp.addData('mouse.x', mouse.x)
thisExp.addData('mouse.y', mouse.y)
thisExp.addData('mouse.time', mouse.time)
# did the participant succeed at the trial?
if correct_counter == click_counter:
    thisExp.addData('practice_heart_success', 'success')
else:
    thisExp.addData('practice_heart_success', 'failure')

# the Routine "train_2" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# ------Prepare to start Routine "start"-------
continueRoutine = True
# update component parameters for each repeat
# setup some python lists for storing info about the mouse_start
mouse_start.clicked_name = []
gotValidClick = False  # until a click is received
text_continue.text = START_TXT_BTN

# fetch audio of read instructions and its duration
read_sound = read_audio_sound_dict['read_start']['sound']
read_dur = read_audio_sound_dict['read_start']['dur']
# start playing instructions
read_sound.play()

# keep track of which components have finished
startComponents = [text_start, mouse_start]
for thisComponent in startComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
startClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "start"-------
while continueRoutine:
    # get current time
    t = startClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=startClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *text_start* updates
    if text_start.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        text_start.frameNStart = frameN  # exact frame index
        text_start.tStart = t  # local t and not account for scr refresh
        text_start.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(text_start, 'tStartRefresh')  # time at next scr refresh
        text_start.setAutoDraw(True)
    # *mouse_start* updates
    if mouse_start.status == NOT_STARTED and t >= read_audio_sound_dict['read_start']['dur']-frameTolerance:
        # keep track of start time/frame for later
        mouse_start.frameNStart = frameN  # exact frame index
        mouse_start.tStart = t  # local t and not account for scr refresh
        mouse_start.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(mouse_start, 'tStartRefresh')  # time at next scr refresh
        mouse_start.status = STARTED
        mouse_start.mouseClock.reset()
        prevButtonState = mouse_start.getPressed()  # if button is down already this ISN'T a new click
    if mouse_start.status == STARTED:  # only update if started and not finished!
        buttons = mouse_start.getPressed()
        if buttons != prevButtonState:  # button state changed?
            prevButtonState = buttons
            if sum(buttons) > 0:  # state changed to a new click
                # check if the mouse was inside our 'clickable' objects
                gotValidClick = False
                try:
                    iter(rectangle_continue)
                    clickableList = rectangle_continue
                except:
                    clickableList = [rectangle_continue]
                for obj in clickableList:
                    if obj.contains(mouse_start):
                        gotValidClick = True
                        mouse_start.clicked_name.append(obj.name)
                if gotValidClick:  
                    continueRoutine = False  # abort routine on response
    # if read instructions audio has finished playing
    if tThisFlip >= read_dur:
        rectangle_continue.draw()
        text_continue.draw()
    
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in startComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "start"-------
for thisComponent in startComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
# store data for thisExp (ExperimentHandler)
thisExp.nextEntry()
text_continue.text = CONTINUE_TXT
# the Routine "start" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# set up handler to look after randomisation of conditions etc
trials = data.TrialHandler(nReps=11, method='random', 
    extraInfo=expInfo, originPath=-1,
    trialList=[None],
    seed=None, name='trials')
thisExp.addLoop(trials)  # add the loop to the experiment
thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
# abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
if thisTrial != None:
    for paramName in thisTrial:
        exec('{} = thisTrial[paramName]'.format(paramName))

for thisTrial in trials:
    currentLoop = trials
    # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
    if thisTrial != None:
        for paramName in thisTrial:
            exec('{} = thisTrial[paramName]'.format(paramName))
    
    # ------Prepare to start Routine "trial_instruction"-------
    continueRoutine = True
    # update component parameters for each repeat
    text_trialinstruction.setText(trialinstruction_mess)
    # reset mouse save data attributes
    mouse.x = []
    mouse.y = []
    mouse.time = []
    mouse.clicked_name = []
    
    # reset counters
    correct_counter = 0
    click_counter = 0
    ellipsis_counter = 0 # for multi-target trials, keeps track of which target is active
    
    # reset target start time keeper
    target_start_time = 0
    
    # reset variable indicating if routine is ready for showing
    # continue button
    ready_continue = False
    
    # reset variable indicating if it's time to switch to the next target
    # image (if this trial has multiple target types)
    next_img_time = False
    
    # reset continue button
    rectangle_continue.opacity = 0
    # empty continue text component's text, which is filled
    # in once participant is allowed to continue
    # (can't simply change opacity of thext text because of a PsychoPy bug)
    text_continue.text = ""
    
    # reset list of cross drawing positions
    cross_position_list.clear()
    
    # clear the list of ellipses stimuli, to make it ready for this trial
    trial_ellipses.clear()
    # specify the context x/y coordinate offsets
    context_y = TRIAL_CONTEXT_Y
    # extract the dictionary of trial information
    trial_dict = trial_ls[trial_counter]
    context_filename = trial_dict['context_fname']
    target_filenames = trial_dict['target_fnames']
    trial_type = trial_dict['type']
    
    # set the context and initial target images
    trial_context_img = context_img_dict[context_filename]['stim']
    trial_target_img =  target_img_dict[target_filenames[0]]
    # extract the resize factor for this routine
    resize_factor = context_img_dict[context_filename]['resize_factor']
    
    # set time limit (based on whether only one type of target is used, or
    # multiple types of targets are used)
    time_limit = SINGLE_TARGET_DUR if len(target_filenames)==1 else PER_TARGET_MAX_DUR
    
    # find which row indices in the ellipses_df that correspond to this trial's
    # target ellipses data
    df_indices = list(ellipses_df[ellipses_df['filename'] == context_filename].index)
    for index in df_indices:
        center_x = px_to_deg(ellipses_df.loc[index, 'center_x']) * resize_factor
        center_y = px_to_deg(ellipses_df.loc[index, 'center_y']) * resize_factor + context_y
        xaxis_r = px_to_deg(ellipses_df.loc[index, 'xaxis_r']) * resize_factor
        yaxis_r = px_to_deg(ellipses_df.loc[index, 'yaxis_r']) * resize_factor
        orientation = ellipses_df.loc[index, 'orientation']
        new_ellipsis = make_pp_ellipsis(
            center_x, 
            center_y, 
            xaxis_r, 
            yaxis_r, 
            orientation
        )
        new_ellipsis.fname = ellipses_df.loc[index, 'target_fname']
        trial_ellipses.append(new_ellipsis)
    
    # make a shallow copy of ellipses list, from which ellipses
    # will be plucked as soon as they are clicked
    nonclicked_ellipses = trial_ellipses[:]
    
    # set instruction message to display
    trialinstruction_mess = type_mess_dict[trial_type]
    text_trialinstruction.setText(trialinstruction_mess)
    
    # keep track of which components have finished
    trial_instructionComponents = [text_trialinstruction]
    for thisComponent in trial_instructionComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    trial_instructionClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "trial_instruction"-------
    while continueRoutine:
        # get current time
        t = trial_instructionClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=trial_instructionClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_trialinstruction* updates
        if text_trialinstruction.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_trialinstruction.frameNStart = frameN  # exact frame index
            text_trialinstruction.tStart = t  # local t and not account for scr refresh
            text_trialinstruction.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_trialinstruction, 'tStartRefresh')  # time at next scr refresh
            text_trialinstruction.setAutoDraw(True)
        if text_trialinstruction.status == STARTED:
            if frameN >= (text_trialinstruction.frameNStart + TRIALINST_DUR):
                # keep track of stop time/frame for later
                text_trialinstruction.tStop = t  # not accounting for scr refresh
                text_trialinstruction.frameNStop = frameN  # exact frame index
                win.timeOnFlip(text_trialinstruction, 'tStopRefresh')  # time at next scr refresh
                text_trialinstruction.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in trial_instructionComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "trial_instruction"-------
    for thisComponent in trial_instructionComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    trials.addData('text_trialinstruction.started', text_trialinstruction.tStartRefresh)
    trials.addData('text_trialinstruction.stopped', text_trialinstruction.tStopRefresh)
    # the Routine "trial_instruction" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # ------Prepare to start Routine "trial"-------
    continueRoutine = True
    # update component parameters for each repeat
    # store start time of trial
    trial_start_time = globalClock.getTime()
    
    # reset mouse click time clock, meaning that mouse click times
    # will be based on __time since trial onset__ (this mouse clock
    # is also used for saving target onset/offset times)
    mouse.mouseClock.reset(0)
    
    if trial_type == 'once':
        # reset 'once' type trial countdown timer (counting in frames) counting down
        # time until next target should be shown
        once_inter_target_counter = ONCE_INTER_TARGET_DUR
        # reset 'once' type trial target onset/offset times lists
        once_target_onset_times = [win.getFutureFlipTime(clock=mouse.mouseClock)]
        once_target_offset_times = []
    
    # keep track of which components have finished
    trialComponents = [text_trial_timekeeper]
    for thisComponent in trialComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    trialClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "trial"-------
    while continueRoutine:
        # get current time
        t = trialClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=trialClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        if not ready_continue:
            trial_context_img.draw()
            trial_target_img.draw()
            rect_trial_target.draw()
            rect_trial_context.draw()
            # draw crosses where participant has clicked
            for position in cross_position_list:
                cross_stimulus.pos = position
                cross_stimulus.draw() 
        
        
        rectangle_continue.draw()
        text_continue.draw()
        
        # note that only data about trial clicks (clicks done before continue
        # button appears, and within context image are) are saved to output 
        # csv file
        buttons = mouse.getPressed()
        if buttons != prevButtonState:  # button state changed?
            prevButtonState = buttons
            # state changed to a new click, and sufficient time has
            # passed since new target presentation
            if sum(buttons) > 0 and ((t - target_start_time) >= BLOCKING_DUR): 
                x, y = mouse.getPos()
                cross_position_list.append((x, y))
                # if continue button is already shown and trial is otherwise
                # over
                if ready_continue:
                    if rectangle_continue.contains(mouse):
                        continueRoutine = False
                # if click within context image area
                elif trial_context_img.contains(mouse): 
                    correct_click = False
                    click_counter += 1
                    # if this trial has multiple targets, each present at only
                    # one location in context image
                    if trial_type == 'once':
                        target_ellipsis = [ell for ell in trial_ellipses if ell.fname == target_filenames[ellipsis_counter]][0]
                        if target_ellipsis.contains(mouse):
                            correct_counter += 1
                            correct_click = True
                            mouse.clicked_name.append(target_ellipsis.fname)
                        # save current time __since trial start__ (hence, using the
                        # 'mouse clock') as target image offset time
                        once_target_offset_times.append(mouse.mouseClock.getTime())
                        next_img_time = True
                    else:
                        # list of indices of ellipses to be removed
                        rm_ell_inds = []
                        # go through all ellipses and check for each one if mouse
                        # was clicked within it
                        for ell_ind in range(len(nonclicked_ellipses)):
                            if nonclicked_ellipses[ell_ind].contains(mouse):
                                correct_counter += 1
                                correct_click = True
                                mouse.clicked_name.append(nonclicked_ellipses[ell_ind].fname)
                                rm_ell_inds.append(ell_ind)
                        # remove clicked ellipses
                        for rm_ell_ind in rm_ell_inds:
                            nonclicked_ellipses.pop(rm_ell_ind)
                            # decrement all 'remove indices' that are higher
                            # than the index whose element was just removed
                            rm_ell_inds = [x-1 if x>rm_ell_ind else x for x in rm_ell_inds]
                    if not correct_click:
                        mouse.clicked_name.append('outside_of_valid_targets')
                    buttons = mouse.getPressed()
                    mouse.time.append(mouse.mouseClock.getTime())
                    mouse.x.append(x)
                    mouse.y.append(y)
        
        if trial_type == 'once':
            # has the response time (minus the time where 
            # the target is hidden) for this target image run out?
            if (t - target_start_time) >= time_limit and not next_img_time:
                # hide the target
                trial_target_img.opacity = 0
                once_inter_target_counter -= 1
            # has response time (including time where target is hidden)
            # for this target image run out?
            if once_inter_target_counter <= 0:
                # save current time __counting from trial onset__ (hence using the
                # 'mouse clock') as target image offset time
                once_target_offset_times.append(mouse.mouseClock.getTime())
                # reset 'once' type trial countdown timer 
                once_inter_target_counter = ONCE_INTER_TARGET_DUR
                next_img_time = True
            if next_img_time:
                # make sure the target is displayed (in case it was 
                # hidden due to trial time running out)
                trial_target_img.opacity = 1
                ellipsis_counter += 1
                if ellipsis_counter >= len(target_filenames):
                    # the trial is considered to be ended as soon as the
                    # continue button appears, hence the time here is
                    # stored as trial end time
                    trial_end_time = globalClock.getTime()
                    ready_continue = True
                    # show continue button
                    rectangle_continue.opacity = 1
                    text_continue.text = CONTINUE_TXT
                else:
                    # save next flip time __counting from trial onset__ (hence using the
                    # 'mouse clock') as target image onset time
                    once_target_onset_times.append(win.getFutureFlipTime(clock=mouse.mouseClock))
                    trial_target_img = target_img_dict[target_filenames[ellipsis_counter]]
                next_img_time = False
                target_start_time = t
        else:
            # has time run out, or all the target areas been clicked?
            if t >= time_limit or len(nonclicked_ellipses)==0:
                if not ready_continue:
                    # the trial is considered to be ended as soon as the
                    # continue button appears, hence the time here is
                    # stored as trial end time
                    trial_end_time = globalClock.getTime()
                    ready_continue = True
                    rectangle_continue.opacity = 1
                    text_continue.text = CONTINUE_TXT
        
        if DEBUG_ON:
            for ell in trial_ellipses:
                ell.draw()
        
        # *text_trial_timekeeper* updates
        if text_trial_timekeeper.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_trial_timekeeper.frameNStart = frameN  # exact frame index
            text_trial_timekeeper.tStart = t  # local t and not account for scr refresh
            text_trial_timekeeper.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_trial_timekeeper, 'tStartRefresh')  # time at next scr refresh
            text_trial_timekeeper.setAutoDraw(True)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in trialComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "trial"-------
    for thisComponent in trialComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    trial_counter += 1
    
    # save trial data
    thisExp.addData('trial_type', trial_type)
    thisExp.addData('context_img_fname', context_filename)
    thisExp.addData('mouse.x', mouse.x)
    thisExp.addData('mouse.y', mouse.y)
    thisExp.addData('mouse.time', mouse.time)
    thisExp.addData('mouse.clicked_target', mouse.clicked_name)
    thisExp.addData('num_correct', correct_counter)
    thisExp.addData('num_clicks', click_counter)
    thisExp.addData('trial_start_time', trial_start_time)
    thisExp.addData('trial_end_time', trial_end_time)
    
    if trial_type == 'once':
        thisExp.addData('once_target_start_times', once_target_onset_times)
        thisExp.addData('once_target_end_times', once_target_offset_times)
    
    # the Routine "trial" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    show_popout_inst = data.TrialHandler(nReps=int(trial_counter == 6), method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=[None],
        seed=None, name='show_popout_inst')
    thisExp.addLoop(show_popout_inst)  # add the loop to the experiment
    thisShow_popout_inst = show_popout_inst.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisShow_popout_inst.rgb)
    if thisShow_popout_inst != None:
        for paramName in thisShow_popout_inst:
            exec('{} = thisShow_popout_inst[paramName]'.format(paramName))
    
    for thisShow_popout_inst in show_popout_inst:
        currentLoop = show_popout_inst
        # abbreviate parameter names if possible (e.g. rgb = thisShow_popout_inst.rgb)
        if thisShow_popout_inst != None:
            for paramName in thisShow_popout_inst:
                exec('{} = thisShow_popout_inst[paramName]'.format(paramName))
        
        # ------Prepare to start Routine "popout_instruction"-------
        continueRoutine = True
        # update component parameters for each repeat
        # here, the value of a supposed 'constant' is actually changed.
        # this really shouldn't be done. however, the 'popout' trials were
        # requested to be added mid-study and little time was given for
        # accomplishing this task. hence, this piece of code is used
        # as a shortcut.
        SINGLE_TARGET_DUR = SINGLE_TARGET_DUR_POPOUT
        sound_popoutinst_readinst.setSound(os.path.join(READ_AUDIO_DIR_PATH, READ_AUDIO_INTRO_FNAME), secs=READ_AUDIO_INTRO_DUR, hamming=True)
        sound_popoutinst_readinst.setVolume(1, log=False)
        # keep track of which components have finished
        popout_instructionComponents = [text_popoutinst, image_showpopout1_nonmarked, image_showpopout1_marked, sound_popoutinst_readinst, image_showpopout2_nonmarked, image_showpopout2_marked, image_showpopout3_nonmarked, image_showpopout3_marked]
        for thisComponent in popout_instructionComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        popout_instructionClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
        frameN = -1
        
        # -------Run Routine "popout_instruction"-------
        while continueRoutine:
            # get current time
            t = popout_instructionClock.getTime()
            tThisFlip = win.getFutureFlipTime(clock=popout_instructionClock)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_popoutinst* updates
            if text_popoutinst.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_popoutinst.frameNStart = frameN  # exact frame index
                text_popoutinst.tStart = t  # local t and not account for scr refresh
                text_popoutinst.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_popoutinst, 'tStartRefresh')  # time at next scr refresh
                text_popoutinst.setAutoDraw(True)
            if text_popoutinst.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_popoutinst.tStartRefresh + INST_SHOWPOPOUT_TXT_DUR-frameTolerance:
                    # keep track of stop time/frame for later
                    text_popoutinst.tStop = t  # not accounting for scr refresh
                    text_popoutinst.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(text_popoutinst, 'tStopRefresh')  # time at next scr refresh
                    text_popoutinst.setAutoDraw(False)
            
            # *image_showpopout1_nonmarked* updates
            if image_showpopout1_nonmarked.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                image_showpopout1_nonmarked.frameNStart = frameN  # exact frame index
                image_showpopout1_nonmarked.tStart = t  # local t and not account for scr refresh
                image_showpopout1_nonmarked.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(image_showpopout1_nonmarked, 'tStartRefresh')  # time at next scr refresh
                image_showpopout1_nonmarked.setAutoDraw(True)
            if image_showpopout1_nonmarked.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > image_showpopout1_nonmarked.tStartRefresh + INST_SHOWPOPOUT_NONMARKED_DUR-frameTolerance:
                    # keep track of stop time/frame for later
                    image_showpopout1_nonmarked.tStop = t  # not accounting for scr refresh
                    image_showpopout1_nonmarked.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(image_showpopout1_nonmarked, 'tStopRefresh')  # time at next scr refresh
                    image_showpopout1_nonmarked.setAutoDraw(False)
            
            # *image_showpopout1_marked* updates
            if image_showpopout1_marked.status == NOT_STARTED and tThisFlip >= INST_SHOWPOPOUT_NONMARKED_DUR-frameTolerance:
                # keep track of start time/frame for later
                image_showpopout1_marked.frameNStart = frameN  # exact frame index
                image_showpopout1_marked.tStart = t  # local t and not account for scr refresh
                image_showpopout1_marked.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(image_showpopout1_marked, 'tStartRefresh')  # time at next scr refresh
                image_showpopout1_marked.setAutoDraw(True)
            if image_showpopout1_marked.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > image_showpopout1_marked.tStartRefresh + INST_SHOWPOPOUT_MARKED_DUR-frameTolerance:
                    # keep track of stop time/frame for later
                    image_showpopout1_marked.tStop = t  # not accounting for scr refresh
                    image_showpopout1_marked.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(image_showpopout1_marked, 'tStopRefresh')  # time at next scr refresh
                    image_showpopout1_marked.setAutoDraw(False)
            # start/stop sound_popoutinst_readinst
            if sound_popoutinst_readinst.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                sound_popoutinst_readinst.frameNStart = frameN  # exact frame index
                sound_popoutinst_readinst.tStart = t  # local t and not account for scr refresh
                sound_popoutinst_readinst.tStartRefresh = tThisFlipGlobal  # on global time
                sound_popoutinst_readinst.play(when=win)  # sync with win flip
            if sound_popoutinst_readinst.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > sound_popoutinst_readinst.tStartRefresh + READ_AUDIO_INTRO_DUR-frameTolerance:
                    # keep track of stop time/frame for later
                    sound_popoutinst_readinst.tStop = t  # not accounting for scr refresh
                    sound_popoutinst_readinst.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(sound_popoutinst_readinst, 'tStopRefresh')  # time at next scr refresh
                    sound_popoutinst_readinst.stop()
            
            # *image_showpopout2_nonmarked* updates
            if image_showpopout2_nonmarked.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                image_showpopout2_nonmarked.frameNStart = frameN  # exact frame index
                image_showpopout2_nonmarked.tStart = t  # local t and not account for scr refresh
                image_showpopout2_nonmarked.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(image_showpopout2_nonmarked, 'tStartRefresh')  # time at next scr refresh
                image_showpopout2_nonmarked.setAutoDraw(True)
            if image_showpopout2_nonmarked.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > image_showpopout2_nonmarked.tStartRefresh + INST_SHOWPOPOUT_NONMARKED_DUR-frameTolerance:
                    # keep track of stop time/frame for later
                    image_showpopout2_nonmarked.tStop = t  # not accounting for scr refresh
                    image_showpopout2_nonmarked.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(image_showpopout2_nonmarked, 'tStopRefresh')  # time at next scr refresh
                    image_showpopout2_nonmarked.setAutoDraw(False)
            
            # *image_showpopout2_marked* updates
            if image_showpopout2_marked.status == NOT_STARTED and tThisFlip >= INST_SHOWPOPOUT_NONMARKED_DUR-frameTolerance:
                # keep track of start time/frame for later
                image_showpopout2_marked.frameNStart = frameN  # exact frame index
                image_showpopout2_marked.tStart = t  # local t and not account for scr refresh
                image_showpopout2_marked.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(image_showpopout2_marked, 'tStartRefresh')  # time at next scr refresh
                image_showpopout2_marked.setAutoDraw(True)
            if image_showpopout2_marked.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > image_showpopout2_marked.tStartRefresh + INST_SHOWPOPOUT_MARKED_DUR-frameTolerance:
                    # keep track of stop time/frame for later
                    image_showpopout2_marked.tStop = t  # not accounting for scr refresh
                    image_showpopout2_marked.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(image_showpopout2_marked, 'tStopRefresh')  # time at next scr refresh
                    image_showpopout2_marked.setAutoDraw(False)
            
            # *image_showpopout3_nonmarked* updates
            if image_showpopout3_nonmarked.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                image_showpopout3_nonmarked.frameNStart = frameN  # exact frame index
                image_showpopout3_nonmarked.tStart = t  # local t and not account for scr refresh
                image_showpopout3_nonmarked.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(image_showpopout3_nonmarked, 'tStartRefresh')  # time at next scr refresh
                image_showpopout3_nonmarked.setAutoDraw(True)
            if image_showpopout3_nonmarked.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > image_showpopout3_nonmarked.tStartRefresh + INST_SHOWPOPOUT_NONMARKED_DUR-frameTolerance:
                    # keep track of stop time/frame for later
                    image_showpopout3_nonmarked.tStop = t  # not accounting for scr refresh
                    image_showpopout3_nonmarked.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(image_showpopout3_nonmarked, 'tStopRefresh')  # time at next scr refresh
                    image_showpopout3_nonmarked.setAutoDraw(False)
            
            # *image_showpopout3_marked* updates
            if image_showpopout3_marked.status == NOT_STARTED and tThisFlip >= INST_SHOWPOPOUT_NONMARKED_DUR-frameTolerance:
                # keep track of start time/frame for later
                image_showpopout3_marked.frameNStart = frameN  # exact frame index
                image_showpopout3_marked.tStart = t  # local t and not account for scr refresh
                image_showpopout3_marked.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(image_showpopout3_marked, 'tStartRefresh')  # time at next scr refresh
                image_showpopout3_marked.setAutoDraw(True)
            if image_showpopout3_marked.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > image_showpopout3_marked.tStartRefresh + INST_SHOWPOPOUT_MARKED_DUR-frameTolerance:
                    # keep track of stop time/frame for later
                    image_showpopout3_marked.tStop = t  # not accounting for scr refresh
                    image_showpopout3_marked.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(image_showpopout3_marked, 'tStopRefresh')  # time at next scr refresh
                    image_showpopout3_marked.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                core.quit()
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in popout_instructionComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # -------Ending Routine "popout_instruction"-------
        for thisComponent in popout_instructionComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        sound_popoutinst_readinst.stop()  # ensure sound has stopped at end of routine
        # the Routine "popout_instruction" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
    # completed int(trial_counter == 6) repeats of 'show_popout_inst'
    
    thisExp.nextEntry()
    
# completed 11 repeats of 'trials'


# ------Prepare to start Routine "end"-------
continueRoutine = True
routineTimer.add(10.000000)
# update component parameters for each repeat
# keep track of which components have finished
endComponents = [text_end]
for thisComponent in endComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
endClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "end"-------
while continueRoutine and routineTimer.getTime() > 0:
    # get current time
    t = endClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=endClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *text_end* updates
    if text_end.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        text_end.frameNStart = frameN  # exact frame index
        text_end.tStart = t  # local t and not account for scr refresh
        text_end.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(text_end, 'tStartRefresh')  # time at next scr refresh
        text_end.setAutoDraw(True)
    if text_end.status == STARTED:
        # is it time to stop? (based on global clock, using actual start)
        if tThisFlipGlobal > text_end.tStartRefresh + 10-frameTolerance:
            # keep track of stop time/frame for later
            text_end.tStop = t  # not accounting for scr refresh
            text_end.frameNStop = frameN  # exact frame index
            win.timeOnFlip(text_end, 'tStopRefresh')  # time at next scr refresh
            text_end.setAutoDraw(False)
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in endComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "end"-------
for thisComponent in endComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
# register that participant has run whole experiment successfully
import csv
testreg_dir_path = os.path.join('..', '..', 'exp_run_registration')
if not os.path.isdir(testreg_dir_path):
    os.mkdir(testreg_dir_path)
testreg_fpath = os.path.join(testreg_dir_path, f"{expInfo['participant']}_expreg.csv")
add_header = not os.path.isfile(testreg_fpath)
with open(testreg_fpath, 'a') as csvfile:
    testreg_writer = csv.writer(csvfile)
    if add_header:
        testreg_writer.writerow([f"{expInfo['participant']}_run_experiments"])
    testreg_writer.writerow([f"{expInfo['expName']}"])


# Flip one final time so any remaining win.callOnFlip() 
# and win.timeOnFlip() tasks get executed before quitting
win.flip()

# these shouldn't be strictly necessary (should auto-save)
thisExp.saveAsWideText(filename+'.csv', delim='auto')
thisExp.saveAsPickle(filename)
logging.flush()
# make sure everything is closed down
if eyetracker:
    eyetracker.setConnectionState(False)
thisExp.abort()  # or data files will save again on exit
win.close()
core.quit()
