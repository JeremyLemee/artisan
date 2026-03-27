import importlib
import sys
import unittest


class ReactPackageInitTests(unittest.TestCase):
    def test_importing_react_does_not_eagerly_import_agent_module(self):
        sys.modules.pop("react", None)
        sys.modules.pop("react.agent", None)

        importlib.import_module("react")

        self.assertIn("react", sys.modules)
        self.assertNotIn("react.agent", sys.modules)


if __name__ == "__main__":
    unittest.main()
