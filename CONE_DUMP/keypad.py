import pygame

import time
# Define some colors.
BLACK = pygame.Color('black')
WHITE = pygame.Color('white')

pygame.init()
done = False

# Used to manage how fast the screen updates.
clock = pygame.time.Clock()

# Initialize the joysticks.
pygame.joystick.init()

while not done:
    time.sleep(0.2)
    # count number of controller
    joystick_count = pygame.joystick.get_count()
    print(joystick_count)
    for i in range(joystick_count):
        joystick = pygame.joystick.Joystick(i)
        joystick.init()
        # Get the name from the OS for the controller/joystick.
        name = joystick.get_name()
        # print(f"Joystick name: {name}")
        print(name)
        for event in pygame.event.get(): # User did something.
            # just controller events
            joyCommand=[pygame.JOYBUTTONDOWN,pygame.JOYBUTTONUP,pygame.JOYAXISMOTION,pygame.JOYHATMOTION]

            # events fired
            if(event.type in joyCommand):            
                buttons = joystick.get_numbuttons()
                
                # single button event
                if event.type == pygame.JOYBUTTONDOWN:
                    for i in range(buttons):

                        # initidate button
                        button = joystick.get_button(i)
                        
                        # convert values between -1 to 1
                        # convert=(float)("{:>2} ".format(button))
                        # if(convert>0):
                        if event.type == pygame.JOYBUTTONDOWN:
                            print("Joystick button pressed.")
                            if(event.button==0):
                                print("A")
                            if(event.button==1):
                                print("B")
                            if(event.button==2):
                                print("X")
                            if(event.button==3):
                                print("Y")
                            if(event.button==4):
                                print("LB")
                            if(event.button==5):
                                print("RB")
                            if(event.button==6):
                                done=True
                                print("Back")
                            if(event.button==7):
                                print("Start")
                            if(event.button==8):
                                print("LSB")
                            if(event.button==9):
                                print("RSB")
                elif event.type == pygame.JOYBUTTONUP:
                    print(event.button)

       
        
        # print ("Joystick name: {}".format(name))

        # Usually axis run in pairs, up/down for one, and left/right for
        # the other.
        axes = joystick.get_numaxes()

        for i in range(axes):
            axis = joystick.get_axis(i)
            
            # convert values between -1 to 1
            convertAxis=(float)("{:>6.3f}".format(axis))
            # print(convertAxis)
            if(convertAxis >= -1 and convertAxis <= 1 and convertAxis !=0):
                if(i==0):
                    print(convertAxis)
                    if(convertAxis<0):
                        print("left")
                    if(convertAxis>0):
                        print("right")
                if(i==1):
                    print(convertAxis)
                    if(convertAxis<0 and i==1):
                        print("up")
                    if(convertAxis>0 and i ==1):
                        print("down")
                if(convertAxis<0 and i==4):
                    print("r left")
                if(convertAxis>0 and i ==4):
                    print("r right")
                if(convertAxis<0 and i==3):
                    print("r up")
                if(convertAxis>0 and i ==3):
                    print("r down")
                if(convertAxis>0 and i ==2):
                    print("LB2")
                if(convertAxis<0 and i ==2):
                    print("RB2")
        
            hats = joystick.get_numhats()             
            for i in range(hats):
                hat = joystick.get_hat(i)
                if(hat[0]==1):
                    print("Hat right")
                if(hat[0]==-1):
                    print("Hat left")
                if(hat[1]==1):
                    print("Hat up")
                if(hat[1]==-1):
                    print("Hat down")
                # print(hat)