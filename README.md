# Billboard-Car-Count (Lightweight Object Detection/Counting)
### Model to count cars in a given area

Notes: I wrote the basic structural code to run from a live video feeds to monitor for traffic buildups during road construction and alert local traffic control services (flaggers). I’m using the code here with their explicit permission, but the base code is now officially owned by Mariposa County (California).

## Loading model onto single board computer (e.g. Raspberry Pi, Beagleboard, ODROID, etc.)
This code has been optimized to work on SBCs or othre low power devices. Given that for most 
personal uses Rasberry Pi will the chosen SBC, I'm going to give a quick into on installing
OpenCV on your Pi (Python is preinstalled in Pi OS).

## Why not TensorFlow if your SBC has a GPU?
This is an option, and will likely become the preffered option in 5-10 years, but if you want
real-time object recognition, TensorFlow is generally to slow to give a reasonable output fps.

## Rasberry Pi version
You shouldn't have to worry too much about this as Python and OpenCV should work similarly
on any Raspberry Pi version past 2 (32-bit operating system). 
Note: Pi Zero requires a bit more fiddling with. You'll have to increase swap space among
a few other details. This is a good guide for Pi Zero:
https://towardsdatascience.com/installing-opencv-in-pizero-w-8e46bd42a3d3

## Installing OpenCV (Steps)
Option 1 (fast): Pip
If you don't already have Pip installed you'll have to add pip through bash with the
folowing command
```
mkdir ~/src && cd ~/src
wget https://bootstrap.pypa.io/get-pip.py
$ sudo python3 get-pip.py
```

Once pip is installed, the following will install OpenCV
```
sudo pip install opencv-contrib-python
```

Option 2 (slow): Compile
I would generally only reccomend this if you plan to do a lot of OpenCV in C++

the following is a good walkthough for compiling:
https://qengineering.eu/install-opencv-4.5-on-raspberry-pi-4.html


## Choosing background subtractor
The choice of background subtractor will have the largest effect on your model.
The following are my general suggestions. Use the most advanced option available.

Very-low CPU:
-Background subtraction - frame difference

Low CPU:
-Background subtraction - mog2

Medium CPU:
-Background subtraction - mog2 + image tuning

High CPU:
-Background subtraction mog2 + image tuning + haar cascade
Note: Layering a haar cascade on top of a backgound subtraction is very finnicky
and requires a lot of tunnig for best results.

Cpu + GPU :
-TensorFlow
Note: Technically, most newer rasberry pi have GPUs, but still have poor perfomance in the
real-world. You should really only use TensorFlow or similar if you have something 
like one of the Nvidia Jetson systems.