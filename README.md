# AlphaAnim
Animations of alpha filtrations of 2D point clouds.  Run alpha.py to execute it.  Code can handle drawing a 2D point cloud as a collection of black pixels in an image

## Dependencies
* gudhi (pip install gudhi)
* persim (pip install persim)

## Examples

### Noisy Circle
<img src = "noisycircle.gif">

### Snowman
<img src = "snowman.gif">

For a user guide call print_user_guide() in a jupyter notebook.
One can customize types of sounds used, using the FM synthesis library from 472 as the current options, the FPS of the final video, the png used, and the number of frames in the video. 

Commands used to construct the final video :
ffmpeg -r 30 -i %d.png -r 30 -qscale 1 out.avi
ffmpeg -i out.avi -i audio.wav -c:v copy -c:a copy final_video.mov

FINAL VIDEO NAME: final_video.mov


NOTE: As we discussed over teams my computer generated this really slow, there are 100 frames and my computer generated one every 7 minutes.....so this has not been tested extensively to be honest. I know it isn't perfect but it didn't seem completely awful to me (at least at 30 fps). 
