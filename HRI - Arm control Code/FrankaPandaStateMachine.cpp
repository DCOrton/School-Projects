// Copyright (c) 2017 Franka Emika GmbH
// Use of this source code is governed by the Apache-2.0 license, see LICENSE

//Modified by DCOH, Nov. 2022

#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>

#include <franka/exception.h>
#include <franka/robot.h>
#include <franka/gripper.h>

#include "examples_common.h"

/**
 * @warning Before executing this example, make sure there is enough space in front and to the side
 * of the robot.
 */

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <robot-hostname>" << std::endl;
    return -1;
  }
  try {
    franka::Robot robot(argv[1]);
    setDefaultBehavior(robot);
    
    // First move the robot to a suitable joint configuration
    std::array<double, 7> q_goal = {{0, -M_PI_4, 0, -3 * M_PI_4, 0, M_PI_2, M_PI_4}};
    MotionGenerator motion_generator(0.5, q_goal);
    std::cout << "WARNING: This example will move the robot! "
              << "Please make sure to have the user stop button at hand!" << std::endl
              << "Press Enter to continue..." << std::endl;
    std::cin.ignore();
    robot.control(motion_generator);
    std::cout << "Finished moving to initial joint configuration." << std::endl;

    // Set additional parameters always before the control loop, NEVER in the control loop!
    // Set collision behavior.
    robot.setCollisionBehavior(
        {{10.0, 10.0, 9.0, 9.0, 8.0, 7.0, 6.0}}, {{10.0, 10.0, 9.0, 9.0, 8.0, 7.0, 6.0}},
        {{10.0, 10.0, 9.0, 9.0, 8.0, 7.0, 6.0}}, {{10.0, 10.0, 9.0, 9.0, 8.0, 7.0, 6.0}},
        {{10.0, 10.0, 10.0, 12.5, 12.5, 12.5}}, {{10.0, 10.0, 10.0, 12.5, 12.5, 12.5}},
        {{10.0, 10.0, 10.0, 12.5, 12.5, 12.5}}, {{10.0, 10.0, 10.0, 12.5, 12.5, 12.5}});
    
    std::uint8_t state = 0; 
    std::uint8_t last_state = 0; 
    
    std::char one_key_press;

    
    while( state != 7 ){
      std::cout << "Press Enter to continue..." << std::endl;
      std::cin.ignore();
    
      switch(state) {
        default:
        case 0:  // Start 
          // Go to ready 
          go_to_ready();
          
          last_state = state; 
          state = 1; 
          break;
        case 1:  // Ready State
          // Ask to grab block 
          std::cout << "Press Enter to continue..." << std::endl;
          std::cin.ignore();
          // Call gripper_close()
          if (!gripper.grasp(grasping_width, 0.1, 60)) {
            std::cout << "Failed to grasp object." << std::endl;
            return -1;
          }
          
          // Ask for right or left or end           
          std::cout << "Select Drop-Off Right (r), Drop-Off Left (l) or End (e)" << std::endl;
          std::cin.get(one_key_press, 1);
          
          if(one_key_press == 'r'){
            // Go Right
            go_to_right() ;
            
            last_state = state; 
            state = 2; 
          }else if(one_key_press == 'l'){
            //Go Left
            go_to_left();
            
            last_state = state; 
            state = 3; 
          }else{
            //End 
            state = 7; 
          }   
          break;
        case 2:  // Drop-off Right          
          // Ask to Drop block 
          std::cout << "Press Enter to continue..." << std::endl;
          std::cin.ignore();
          gripper.stop();
          
          //Go back to ready
          go_to_ready();
                    
          last_state = state; 
          state = 1; 
          break;
        case 3:  // Drop-off Right
          // Ask to Drop block 
          std::cout << "Press Enter to continue..." << std::endl;
          std::cin.ignore();
          gripper.stop();
          
          //Go back to ready
          go_to_ready();
                    
          last_state = state; 
          state = 1;           
          break;
      } 
    }
    return 0;
}
    
void go_to_ready() {
 //TODO:
}
void go_to_right() {
  //TODO:
}  
void go_to_left() {
  //TODO:
}  
