"""The live runtime: run a policy on your machine against the Brain simulator.

The simulator stays in the Brain page; each control step it sends the robot's
observation here and waits for the action (lockstep - a slow model makes the
run slower, never wrong). Both ends connect OUT to the platform's relay, so
there is nothing to open on your firewall. Protocol: docs/protocol.md.
"""
