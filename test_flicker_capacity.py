from rcql_advanced_tests import run_flicker_catch
import fileinput
import sys

# Temporarily modify rcql_advanced_tests.py
for line in fileinput.input('rcql_advanced_tests.py', inplace=True):
    if 'max_total_transitions=num_episodes' in line:
        print(line.replace('max_total_transitions=num_episodes', 'max_total_transitions=20000'), end='')
    else:
        print(line, end='')

run_flicker_catch(1000)
