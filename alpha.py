"""
Programmer: Chris Tralie
Purpose: To animate an alpha filtration
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters
from scipy import spatial
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import IPython.display as ipd
from instruments import * 

from persim import plot_diagrams
import gudhi

def print_user_guide():
    
    print('Welcome to the Alpha Animation with Sound python program!')
    current_sounds = get_sound_dict().keys()
    print()
    print('The current supported sounds are the following')
    print('----------------------------------------------')
    sounds = get_sound_dict()
    for sound in sounds.keys():
        print(sound)
    print()
    print('You can add sound to the following events: the addition of an edge, the creation of a triangle, and the birth and death of topological events')
    print('You should also specificy what you FPS you intend to make the final animation so the audio lines up. The default is 30 frames per second.')
    print('Start the program by calling the run_alpha() method. Make sure to specificy the png of your point cloud. You can add sounds through this method as well')
    print()
    print('See an example run below')
    print('------------------------')
    print('run_alpha(png="snowman.png",fps=30,triangle_sound="brass",edge_sound="dirty bass",birth_sound="wood",death_sound="bell")')
    print()
    print('You can also specify the number of frames in the final product with the scales argument')
    print('Also note that is not required to specify all sounds (or any sounds). If one or any sounds are not declared, the program will pick which sounds to use for each event.')
    
    
def get_sound_dict():
    
    dictionary = dict({'karplus': karplus_strong_note,'plucked string':fm_plucked_string_note,'bell':fm_bell_note,'brass':fm_brass_note,'drum':fm_drum_sound,'dirty bass':fm_dirty_bass_note,'wood':fm_wood_drum_sound})
    
    return dictionary

def set_sound(event,mem):
    """
    Sets which sound to be played for a specific event as requested by user 
    
    Parameters
    ----------
    event: String
        Type of event (triangle being added for example)
    Mem: List
        Used only if a sound was not specified by the user for this event. Mem olds which sounds have already been chosen and chooses a different one for this event. Will start repeating if program runs out of sounds (which would only happen if more events were added but robustness is important :) )
        
    Returns
    -------
    ret: Method name for sound. 
    mem: Used for next event 
    """
    
    sound_dict = get_sound_dict()
    
    ret = ''
    
    
    if len(event) == 0:

        #so sorry about this lil section 
       
        #loop over sounds until the program finds one that has not been used yet
        for sound in sound_dict.keys():
            if sound not in mem:
                ret = sound_dict[sound]
                mem.append(sound)
                break
    else: 
        mem.append(event)
        ret = sound_dict[event]

    return ret,mem

def make_audio_array(scales,filtration,dgmsalpha,fps=30,triangle_sound='',edge_sound='',birth_sound ='',death_sound=''):
    """ 
    Make the audio array for the animation 
    
    Parameters
    ----------
    
    scales: NP ARRAY, Scales to use in each frame of the animation.
    filtration: NP ARRAY, edges and triangles and the alphas at which they are added
    dgmsalpha: NP ARRAY, birth and death alphas for topological events 
    fps: int, frames per second the animation is run at 
    triangle_sound: String, type of sound to play when a triangle is added
    edge_sound: String, type of sound to play when an edge is added 
    birth_sound: String, type of sound to play when a topoligical event is born 
    death_sound: String, type of sound to play when a topoligical event dies (rip)
     
    Returns
    -------
    
    audio: NP ARRAY. array to be converted to audio. 
    sr: INT, sample rate to play audio at
    """
    max_alpha = max(scales)
    filtration = np.array(filtration)
    dgmsalpha = np.array(dgmsalpha)
    
    #cut out anything above the maximum alpha value in the animation
    filtration = filtration[np.where(filtration[:,1] <= max_alpha)]
  #  dgmsalpha = dgmsalpha[np.where(dgmsalpha[0,:] <= max_alpha)]
  #  dgmsalpha = dgmsalpha[np.where(dgmsalpha[:,1] <= max_alpha)]
    
    
    N = len(filtration) + len(dgmsalpha)
    scale = len(scales)
    audio = np.zeros(N*scale)
    mem = []
    
    #set which sound to play for each event
    triangle_sound,mem = set_sound(triangle_sound,mem)
    edge_sound,mem = set_sound(edge_sound,mem)
    birth_sound,mem = set_sound(birth_sound,mem)
    death_sound,mem = set_sound(death_sound,mem)
   
    
    sr = round((200*scale)*fps/scale)
    #compensate for the number of the frames
    
    
    frame_comp = fps * scale
    
    rate = sr
    dur = 1
    samples = len(audio)
    
    
    for i in dgmsalpha:
        birth = i[0]
        death = i[1]
        d = {'birth':birth,'death':death}
        for event,alpha in d.items():
        
              #start = int(round(scale/alpha+(alpha/max_alpha)*samples))
              #keeping this in just so you can see how wrong i was at first HAHHAHAHA 
        
        
            start = int(round((alpha/max_alpha)*samples))
            end = int(round(sr+start))
            if end < samples:
                if event ==  'birth':
                    audio[start:end] = birth_sound(sr=rate,duration=dur)                
                elif event == 'death':
                    audio[start:end] = death_sound(sr=rate,duration=dur)
                
    print(scale)
    for i in filtration:
        event = i[0]
        alpha = i[1]
        start = int(round((alpha/max_alpha)*samples))
        end = int(round(sr+start))
        if end < samples:
            if len(event) > 2: 
                audio[start:end] = triangle_sound(sr=rate,duration=dur)           
            elif len(event) == 2:
                audio[start:end] = edge_sound(sr=rate,duration=dur)  
            
    return audio,sr
        
def gudhi2persim(pers):
    """
    Convert a persistence diagram from GUDHI's format into
    a format that persim can plot
    """
    Is = [[], []]
    for i in range(len(pers)):
        (dim, (b, d)) = pers[i]
        Is[dim].append([b, d])
    #Put onto diameter scale so it matches rips more closely
    return [np.sqrt(np.array(I)) for I in Is]


def draw_alpha(X, filtration, alpha, draw_balls=True, draw_voronoi_edges=True):
    """
    Draw the delaunay triangulation in dotted lines, with the alpha faces at
    a particular scale

    Parameters
    ----------
    X: ndarray(N, 2)
        A 2D point cloud
    filtration: list of [(idxs, d)]
        List of simplices in the filtration, listed by idxs, which indexes into
        X, and with an associated scale d at which the simplex enters the filtration
    alpha: float
        The radius/scale up to which to plot balls/simplices
    draw_balls: boolean
        Whether to draw the balls (discs intersected with voronoi regions)
    draw_voronoi_edges: boolean
        Whether to draw the voronoi edges showing the boundaries of the alpha balls
    """
    
    # Determine limits of plot
    pad = 0.3
    xlims = [np.min(X[:, 0]), np.max(X[:, 0])]
    xr = xlims[1]-xlims[0]
    ylims = [np.min(X[:, 1]), np.max(X[:, 1])]
    yr = ylims[1]-ylims[0]
    xlims[0] -= xr*pad
    xlims[1] += xr*pad
    ylims[0] -= yr*pad
    ylims[1] += yr*pad

    if draw_balls:
        resol = 2000
        xr = np.linspace(xlims[0], xlims[1], resol)
        yr = np.linspace(ylims[0], ylims[1], resol)
        xpix, ypix = np.meshgrid(xr, yr)
        P = np.ones((xpix.shape[0], xpix.shape[1], 4))
        PComponent = np.ones_like(xpix)
        PBound = np.zeros_like(PComponent)
        # First make balls
        tree = spatial.KDTree(X)
        XPix = np.array([xpix.flatten(), ypix.flatten()]).T
        neighbs = tree.query(XPix, 1)[1].flatten()
        neighbs = np.reshape(neighbs, xpix.shape)
        if draw_voronoi_edges:
            PBound = filters.sobel(neighbs) > 0
        else:
            PBound = np.zeros_like(neighbs)
        for i in range(X.shape[0]):
            # First make the ball part
            ballPart = (xpix-X[i, 0])**2 + (ypix-X[i, 1])**2 <= alpha**2
            # Now make the Voronoi part
            voronoiPart = np.reshape(neighbs == i, ballPart.shape)
            Pi = ballPart*voronoiPart
            PComponent[Pi == 1] = 0
        # Now make Voronoi regions
        P[:, :, 0] = PComponent
        P[:, :, 1] = PComponent
        P[:, :, 3] = 0.2 + 0.8*PBound
        plt.imshow(np.flipud(P), cmap='magma', extent=(xlims[0], xlims[1], ylims[0], ylims[1]))

    # Plot simplices
    patches = []
    for (idxs, d) in filtration:
        if len(idxs) == 2:
            if d < alpha:
                plt.plot(X[idxs, 0], X[idxs, 1], 'k', 2)
            else:
                plt.plot(X[idxs, 0], X[idxs, 1], 'gray', linestyle='--', linewidth=1)
        elif len(idxs) == 3 and d < alpha:
            patches.append(Polygon(X[idxs, :]))
    ax = plt.gca()
    p = PatchCollection(patches, alpha=0.2, facecolors='C1')
    ax.add_collection(p)
    plt.scatter(X[:, 0], X[:, 1], zorder=0)
    plt.xlim(xlims[0], xlims[1])
    plt.ylim(ylims[0], ylims[1])
    
    #plt.axis('equal')


def alpha_animation(X, fps,triangle_sound,edge_sound,birth_sound,death_sound,scales):
    """
    Create an animation of an alpha filtration of a 2D point cloud
    with the point cloud on the left and a plot on the right

    Parameters
    ----------
    X: ndarray(N, 2)
        A point cloud
    scales: ndarray(T)
        Scales to use in each frame of the animation.  If left blank,
        the program will choose 100 uniformly spaced scales between
        the minimum and maximum events in birth/death
    """
  
    alpha_complex = gudhi.AlphaComplex(points=X)
    simplex_tree = alpha_complex.create_simplex_tree()
    filtration = [(f[0], np.sqrt(f[1])) for f in simplex_tree.get_filtration()]
    diag = simplex_tree.persistence()
    dgmsalpha = gudhi2persim(diag)[0:2]
    
    
    if scales.size == 0:
        # Choose some default scales based on persistence
        smin = min(np.min(dgmsalpha[0]), np.min(dgmsalpha[1]))
        smax = max(np.max(dgmsalpha[0][np.isfinite(dgmsalpha[0])]), np.max(dgmsalpha[1]))
        rg = smax-smin
        smin = max(0, smin-0.05*rg)
        smax += 0.05*rg
        scales = np.linspace(smin, smax, 100)

    audio =  make_audio_array(scales,filtration,dgmsalpha[1],fps,triangle_sound,edge_sound,birth_sound,death_sound)
 #   plt.figure(figsize=(12, 6))
#    for frame, alpha in enumerate(scales):
 #       plt.clf()
  #      plt.subplot(121)
   #     draw_alpha(X, filtration, alpha, True)
    #    plt.title("$\\alpha = {:.3f}$".format(alpha))
     #   plt.subplot(122)
      #  plot_diagrams(dgmsalpha)
       # plt.plot([-0.01, alpha], [alpha, alpha], 'gray', linestyle='--', linewidth=1, zorder=0)
       # plt.plot([alpha, alpha], [alpha, 1.0], 'gray', linestyle='--', linewidth=1, zorder=0)
      #  plt.text(alpha+0.01, alpha-0.01, "{:.3f}".format(alpha))
     #   plt.title("Persistence Diagram")
    #    plt.savefig("{}.png".format(frame), bbox_inches='tight')


    return audio
def load_pointcloud(path):
    """
    Load a point cloud from an image.  Every black pixel
    is considered to represent a point

    Parameters
    ----------
    path: string
        Path to point cloud
    
    Returns
    -------
    ndarray(N, 2)
        A point cloud with N points
    """
    from skimage.io import imread
    I = imread(path)
    I = np.mean(I, axis=2)
    X, Y = np.meshgrid(np.arange(I.shape[1]), np.arange(I.shape[1]))
    x = X[I < 255]
    y = I.shape[0]-Y[I <255]
    return np.array([x, y]).T

def get_noisy_circle():
    """
    An example of a simple noisy circle with 20 points
    """
    np.random.seed(0)
    X = np.random.randn(20, 2)
    X /= np.sqrt(np.sum(X**2, 1))[:, None]
    X += 0.2*np.random.randn(X.shape[0], 2)
    return X

def run_alpha(png,scales=np.array([]),fps=30,triangle_sound='',edge_sound='',birth_sound ='' ,death_sound=''):
    X = load_pointcloud(png)
    
    #check if supported sound type
    for event in [triangle_sound,edge_sound,birth_sound,death_sound]:
        if len(event) != 0:
            sound_dict = get_sound_dict()
            if event not in sound_dict.keys():
                    raise Exception(f'Sorry {event} is not a supported sound....yet :)')
                
    # If scales left empty, will choose 100 automatically
    # based on persistence diagrams
   
   # scales = np.linspace(3, 18, 500) # Choose custom scales
    audio,sr = alpha_animation(X,fps,triangle_sound,edge_sound,birth_sound,death_sound,scales)
    return audio,sr