cmake_minimum_required(VERSION 3.0.2)
project(challenge_cup_task_template)

find_package(catkin REQUIRED)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/challenge_task.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
