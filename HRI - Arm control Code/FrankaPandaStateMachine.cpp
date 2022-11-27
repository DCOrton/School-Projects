
// DCOH, Nov. 2022

#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <char>


void go_to_ready() {
 //TODO:
}
void go_to_right() {
  //TODO:
}  
void go_to_left() {
  //TODO:
}  

int main() {   
    std::uint8_t state = 0; 
    std::uint8_t last_state = 0; 
    
    std::char one_key_press = '';

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
    

