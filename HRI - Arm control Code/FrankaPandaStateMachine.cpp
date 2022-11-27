// Copyright (c) 2017 Franka Emika GmbH
// Use of this source code is governed by the Apache-2.0 license, see LICENSE
#include <cmath>
#include <iostream>

#include <franka/exception.h>
#include <franka/robot.h>

#include "examples_common.h"

/**
 * @example generate_consecutive_motions.cpp
 * An example showing how to execute consecutive motions with error recovery.
 *
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

    for (size_t i = 0; i < 5; i++) {
      std::cout << "Executing motion." << std::endl;
      try {
        double time_max = 4.0;
        double omega_max = 0.2;
        double time = 0.0;
        robot.control([=, &time](const franka::RobotState&,
                                 franka::Duration period) -> franka::JointVelocities {
          time += period.toSec();

          double cycle = std::floor(std::pow(-1.0, (time - std::fmod(time, time_max)) / time_max));
          double omega = cycle * omega_max / 2.0 * (1.0 - std::cos(2.0 * M_PI / time_max * time));

          franka::JointVelocities velocities = {{0.0, 0.0, omega, 0.0, 0.0, 0.0, 0.0}};
          if (time >= 2 * time_max) {
            std::cout << std::endl << "Finished motion." << std::endl;
            return franka::MotionFinished(velocities);
          }
          return velocities;
        });
      } catch (const franka::ControlException& e) {
        std::cout << e.what() << std::endl;
        std::cout << "Running error recovery..." << std::endl;
        robot.automaticErrorRecovery();
      }
    }
  } catch (const franka::Exception& e) {
    std::cout << e.what() << std::endl;
    return -1;
  }

  std::cout << "Finished." << std::endl;

  return 0;
}






// Copyright (c) 2017 Franka Emika GmbH
// Use of this source code is governed by the Apache-2.0 license, see LICENSE
#include <iostream>
#include <sstream>
#include <string>
#include <thread>

#include <franka/exception.h>
#include <franka/gripper.h>

/**
 * @example grasp_object.cpp
 * An example showing how to control FRANKA's gripper.
 */

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr << "Usage: ./grasp_object <gripper-hostname> <homing> <object-width>" << std::endl;
    return -1;
  }

  try {
    franka::Gripper gripper(argv[1]);
    double grasping_width = std::stod(argv[3]);

    std::stringstream ss(argv[2]);
    bool homing;
    if (!(ss >> homing)) {
      std::cerr << "<homing> can be 0 or 1." << std::endl;
      return -1;
    }

    if (homing) {
      // Do a homing in order to estimate the maximum grasping width with the current fingers.
      gripper.homing();
    }

    // Check for the maximum grasping width.
    franka::GripperState gripper_state = gripper.readOnce();
    if (gripper_state.max_width < grasping_width) {
      std::cout << "Object is too large for the current fingers on the gripper." << std::endl;
      return -1;
    }

    // Grasp the object.
    if (!gripper.grasp(grasping_width, 0.1, 60)) {
      std::cout << "Failed to grasp object." << std::endl;
      return -1;
    }

    // Wait 3s and check afterwards, if the object is still grasped.
    std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(3000));

    gripper_state = gripper.readOnce();
    if (!gripper_state.is_grasped) {
      std::cout << "Object lost." << std::endl;
      return -1;
    }

    std::cout << "Grasped object, will release it now." << std::endl;
    gripper.stop();
  } catch (franka::Exception const& e) {
    std::cout << e.what() << std::endl;
    return -1;
  }

  return 0;
}
