"""MOLOCH State-Engine-Authority (PC-Side, Welle DH-6).

Pi-Opus' state_vector.py ist Lightweight Reflector mit apply_pc_authority(vector)-API.
PC haelt die volle Transition-Engine + Safety + Logger und schreibt
authoritative State-Vector zurueck an Pi.

Integration in pc/state_aggregator.py: nach EMA-Update wird der Vector
durch transition_engine + safety_layer geleitet, geloggt und (wenn endpoint live)
an Pi gepostet.
"""
