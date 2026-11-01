"""
Tests for the creation name that pytta.generate and the measurement run
methods give to the objects they return: the name assigned on the
module-level line of the calling script.

The name is read from the call stack, so the tests run real scripts in a
subprocess: under pytest the stack is deep and hides failures that a plain
script hits (Python 3.12 changed where traceback.walk_stack(None) starts).
"""
import os
import subprocess
import sys
import tempfile
import unittest

ENV = dict(os.environ, MPLBACKEND='Agg')


def run(*args):
    out = subprocess.run([sys.executable, *args], capture_output=True,
                         text=True, env=ENV)
    return out.returncode, out.stdout.strip(), out.stderr


class TestCreationName(unittest.TestCase):

    def test_script_gets_the_module_level_name(self):
        # At module level the name is the assigned variable. Inside a
        # function the generator still works; the name comes from the
        # module-level line that led to the call.
        script = ('import pytta\n'
                  'x = pytta.generate.sweep()\n'
                  'print(x.creation_name)\n'
                  'def main():\n'
                  '    y = pytta.generate.sin()\n'
                  '    print(y.creation_name)\n'
                  'main()\n')
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'script.py')
            with open(path, 'w') as f:
                f.write(script)
            code, out, err = run(path)
        self.assertEqual(code, 0, err)
        self.assertEqual(out.splitlines(), ['x', 'main()'])

    def test_works_without_source_lines(self):
        # Under `python -c` the source of the calling line is not
        # available: the object is created with an empty name.
        code, out, err = run('-c', 'import pytta; x = pytta.generate.sweep(); '
                                   'print(repr(x.creation_name))')
        self.assertEqual(code, 0, err)
        self.assertEqual(out, "''")


if __name__ == '__main__':
    unittest.main()
