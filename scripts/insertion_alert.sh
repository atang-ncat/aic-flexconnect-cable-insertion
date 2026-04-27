#!/bin/bash
ros2 topic echo /scoring/insertion_event | while read line; do
  if [[ "$line" == *"sc_port"* || "$line" == *"sfp_port"* ]]; then
    echo -e "\a\a\a\a\a"
    echo ">>> INSERTED! Press Right Arrow <<<"
  fi
done
