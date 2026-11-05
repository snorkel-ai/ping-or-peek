#!/bin/sh
# Can the agent under test read the simulated user's prompt?
#
# Run inside the trial container with the agent's privileges while the user agent is live
# (docker exec, or ask the agent to run it). It searches everything readable for a sentinel
# that exists only in the user's prompt, and prints where it was found, never the content.
# The sentinel is assembled at run time, so this script and its command line never contain it.
P="$(printf '%s-%s-%s' "${1:-ZQX7}" PERSONA "${2:-K4W2}")"

echo "== text files under / (grep -rIl, excluding /proc /sys /dev)"
grep -rIl --exclude-dir=proc --exclude-dir=sys --exclude-dir=dev -e "$P" / 2>/dev/null

echo "== any files, binary included, in home, temp and log directories"
grep -ral -e "$P" /root /home /tmp /var/tmp /run /logs /workspace /installed-agent 2>/dev/null

echo "== process environments and command lines"
for d in /proc/[0-9]*; do
  for f in environ cmdline; do
    if grep -qa -e "$P" "$d/$f" 2>/dev/null; then
      echo "$d/$f (process: $(cat "$d/comm" 2>/dev/null))"
    fi
  done
done

echo "== this shell's environment"
echo "env matches: $(env | grep -c -e "$P")"

echo "== tmux panes"
for s in $(tmux ls -F '#S' 2>/dev/null); do
  echo "tmux pane $s matches: $(tmux capture-pane -p -S - -t "$s" 2>/dev/null | grep -c -e "$P")"
done

echo "== done"
