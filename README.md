# Team 3: Foraging and Flocking

*This package assumes that you have the miro-docker container installed and working.
If you don't, you can install it [here](https://github.com/AlexandrLucas/miro-docker).*

### Cloning our repository

Run the miro-docker container with `miro-hub start`  
Attach a shell with `miro-hub term`

From the root in the miro-docker container:
```
cd mdk/catkin_ws/src
```
then clone our repository with:
```
git clone https://github.com/lewislow22/COM3528-TEAM3-flocking.git
```

### Building our package

Go back to the catkin_ws directory (```cd ..```)  
Build our package:
```
catkin build
```
Re-source:
```
source ~/.bashrc
```

### Running our program

You can now run our program with:
```
roslaunch com3528_flocking forage.launch
```

## Enjoy!!